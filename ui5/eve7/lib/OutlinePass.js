/**
 * @author spidersharma / http://eduperiment.com/
 */

THREE.OutlinePass = function ( resolution, scene, camera ) {

	// [{ "index": number, "isPoints": boolean, "pointSize": number, "vertShader": string, "fragShader":string },......]
	this.renderScene = scene;
	this.renderCamera = camera;

	// [fElementId][elementId] -> { "sel_type": THREE.OutlinePass.selection_enum, "sec_sel": boolean, "geom": Primitive<> }
	this.id2obj_map = {};
	// R: Primitives
	this._selectedObjects = [];
	// [C1]: Selection Types - [C2]: Attributes(color, size, etc...) - [R2]: Primitives
	this._groups = Array.from(Array(THREE.OutlinePass.selection_enum.total), () => []); // ES6 (could be replaced with vanilla Js)

	this.edgeGlow = 0.0;
	this.usePatternTexture = false;
	this.edgeThickness = 1.0;
	this.edgeStrength = 3.0;
	this.downSampleRatio = 2;
	this.glowDownSampleRatio = 2;

	THREE.Pass.call( this );

	this.resolution = ( resolution !== undefined ) ? new THREE.Vector2( resolution.x, resolution.y ) : new THREE.Vector2( 256, 256 );

	var pars = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RGBAFormat };

	var resx = Math.round( this.resolution.x / this.downSampleRatio );
	var resy = Math.round( this.resolution.y / this.downSampleRatio );

	this.maskBufferMaterial = new THREE.MeshBasicMaterial( { color: 0xffffff } );
	this.maskBufferMaterial.side = THREE.DoubleSide;

	this.renderTargetMaskBuffer = [];
	// +1 extra for the "accumulated" result
	for(let i = 0; i < THREE.OutlinePass.selection_enum.total; ++i){
		let maskBuffer = new THREE.WebGLRenderTarget( this.resolution.x, this.resolution.y, pars )
		maskBuffer.texture.name = "OutlinePass.submask"+i;
		maskBuffer.texture.generateMipmaps = false;
		
		this.renderTargetMaskBuffer.push(maskBuffer);
	}

	this.renderTargetMaskBufferMain = new THREE.WebGLRenderTarget( this.resolution.x, this.resolution.y, pars );
	this.renderTargetMaskBufferMain.texture.name = "OutlinePass.mask";
	this.renderTargetMaskBufferMain.texture.generateMipmaps = false;

	this.depthMaterial = new THREE.MeshDepthMaterial();
	this.depthMaterial.side = THREE.DoubleSide;
	this.depthMaterial.depthPacking = THREE.RGBADepthPacking;
	this.depthMaterial.blending = THREE.NoBlending;

	this.prepareMaskMaterial = this.getPrepareMaskMaterial();
	this.prepareMaskMaterial.side = THREE.DoubleSide;
	this.prepareMaskMaterial.fragmentShader = replaceDepthToViewZ( this.prepareMaskMaterial.fragmentShader, this.renderCamera );

	this.renderTargetDepthBuffer = new THREE.WebGLRenderTarget( this.resolution.x, this.resolution.y, pars );
	this.renderTargetDepthBuffer.texture.name = "OutlinePass.depth";
	this.renderTargetDepthBuffer.texture.generateMipmaps = false;

	this.renderTargetMaskDownSampleBuffer = new THREE.WebGLRenderTarget( resx, resy, pars );
	this.renderTargetMaskDownSampleBuffer.texture.name = "OutlinePass.depthDownSample";
	this.renderTargetMaskDownSampleBuffer.texture.generateMipmaps = false;

	this.renderTargetBlurBuffer1 = new THREE.WebGLRenderTarget( resx, resy, pars );
	this.renderTargetBlurBuffer1.texture.name = "OutlinePass.blur1";
	this.renderTargetBlurBuffer1.texture.generateMipmaps = false;
	this.renderTargetBlurBuffer2 = new THREE.WebGLRenderTarget( Math.round( resx / this.glowDownSampleRatio ), Math.round( resy / this.glowDownSampleRatio ), pars );
	this.renderTargetBlurBuffer2.texture.name = "OutlinePass.blur2";
	this.renderTargetBlurBuffer2.texture.generateMipmaps = false;

	this.edgeDetectionMaterial = this.getEdgeDetectionMaterial();
	this.renderTargetEdgeBuffer1 = new THREE.WebGLRenderTarget( resx, resy, pars );
	this.renderTargetEdgeBuffer1.texture.name = "OutlinePass.edge1";
	this.renderTargetEdgeBuffer1.texture.generateMipmaps = false;
	this.renderTargetEdgeBuffer2 = new THREE.WebGLRenderTarget( Math.round( resx / this.glowDownSampleRatio ), Math.round( resy / this.glowDownSampleRatio ), pars );
	this.renderTargetEdgeBuffer2.texture.name = "OutlinePass.edge2";
	this.renderTargetEdgeBuffer2.texture.generateMipmaps = false;

	const MAX_EDGE_THICKNESS = 4;
	const MAX_EDGE_GLOW = 4;

	this.separableBlurMaterial1 = this.getSeperableBlurMaterial( MAX_EDGE_THICKNESS );
	this.separableBlurMaterial1.uniforms[ "texSize" ].value = new THREE.Vector2( resx, resy );
	this.separableBlurMaterial1.uniforms[ "kernelRadius" ].value = 1;
	this.separableBlurMaterial2 = this.getSeperableBlurMaterial( MAX_EDGE_GLOW );
	this.separableBlurMaterial2.uniforms[ "texSize" ].value = new THREE.Vector2( Math.round( resx / this.glowDownSampleRatio ), Math.round( resy / this.glowDownSampleRatio ) );
	this.separableBlurMaterial2.uniforms[ "kernelRadius" ].value = MAX_EDGE_GLOW;

	// Overlay material
	this.overlayMaterial = this.getOverlayMaterial();

	// copy material
	if ( THREE.CopyShader === undefined )
		console.error( "THREE.OutlinePass relies on THREE.CopyShader" );

	var copyShader = THREE.CopyShader;

	this.copyUniforms = THREE.UniformsUtils.clone( copyShader.uniforms );
	this.copyUniforms[ "opacity" ].value = 1.0;

	this.materialCopy = new THREE.ShaderMaterial( {
		uniforms: this.copyUniforms,
		vertexShader: copyShader.vertexShader,
		fragmentShader: copyShader.fragmentShader,
		blending: THREE.NoBlending,
		depthTest: false,
		depthWrite: false,
		transparent: true
	} );

	this.enabled = true;
	this.needsSwap = false;

	this.oldClearColor = new THREE.Color();
	this.oldClearAlpha = 1;

	this.camera = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 1 );
	this.scene = new THREE.Scene();

	this.quad = new THREE.Mesh( new THREE.PlaneBufferGeometry( 2, 2 ), null );
	this.quad.frustumCulled = false; // Avoid getting clipped
	this.scene.add( this.quad );

	this.tempPulseColor1 = new THREE.Color();
	this.tempPulseColor2 = new THREE.Color();
	this.textureMatrix = new THREE.Matrix4();

	function replaceDepthToViewZ( string, camera ) {

		var type = camera.isPerspectiveCamera ? 'perspective' : 'orthographic';

		return string.replace( /DEPTH_TO_VIEW_Z/g, type + 'DepthToViewZ' );

	}

};

THREE.OutlinePass.prototype = Object.assign( Object.create( THREE.Pass.prototype ), {

	constructor: THREE.OutlinePass,

	parseAtts: function(object, groups){
		let found = false;
		// loop over groups
		for (let z = 0; z < groups.length; ++z){
			// loop over all the elements of a group
			//for (let w = 0; w < groups[z].length; ++w)

			const w = 0; // check only the first element of the group!
			{
				let pass = (object.type === "Points") ? (groups[z][w].material.size === object.material.size) : true;
				
				if( pass ){
					// final check
					if( groups[z][w]["vertShader"] === object["vertShader"]	&&
						groups[z][w]["fragShader"] === object["fragShader"] &&
						groups[z][w].visibleEdgeColor === object.visibleEdgeColor &&
						groups[z][w].hiddenEdgeColor === object.hiddenEdgeColor
					) { 
						groups[z].push(object);
						found = true;
						break; 
					}
				}
			}
			if(found)
				break;
		}

		if(!found){
			groups.push([object]);
		}
	},

	checkForCustomAtts: function(){
		/*
		this.atts = {
			"index": [],
			"pointSize": [],
			"vertShader": [],
			"fragShader": [],
			"total": 0
		}; // reset
		for (let i = 0; i < this.selectedObjects.length; ++i){
			const object = this.selectedObjects[i];

			if(object.type === "Points"){
				this.atts["index"].push(i);
				this.atts["pointSize"].push(object.material.size || 20.0);
				this.atts["vertShader"].push(object["vertShader"]);
				this.atts["fragShader"].push(object["fragShader"]);
				++this.atts.total;
			}
		}
		*/
		
		// one array for each selection type
		this._groups = Array.from(Array(THREE.OutlinePass.selection_enum.total), () => []);

		// fill in "this._groups"
		for (const obj of this._selectedObjects){
			for(const geom of obj.geom){
				this.parseAtts(geom, this._groups[obj.sel_type]);
			}
		}

		// for (let i = 0; i < this.selectedObjects.length; ++i){
		// 	const object = this.selectedObjects[i];

		// 	// treat Mesh and LineSegments as the same
		// 	if(object.type === "Mesh" || object.type === "LineSegments" )
		// 	{
		// 		groups[0] = groups[0] || [];
		// 		groups[0].push(object);
		// 	}
		// 	else if(object.type === "Points")
		// 	{
		// 		let found = false;
		// 		// loop over groups
		// 		for (let z = 1; z < groups.length; ++z){
		// 			// loop over all the elements of a group
		// 			for (let w = 0; w < z.length; ++w){
		// 				// if the objects have the same attributes
		// 				if(
		// 					this.selectedObjects[z][w].type			 === object.type			&&
		// 					this.selectedObjects[z][w].material.size === object.material.size	&&
		// 					this.selectedObjects[z][w]["vertShader"] === object["vertShader"]	&&
		// 					this.selectedObjects[z][w]["fragShader"] === object["fragShader"]
		// 				){
		// 					groups[z].push(object);
		// 					found = true;
		// 					break; 
		// 				}
		// 			}
		// 			if(found)
		// 				break;
		// 		}

		// 		if(!found){
		// 			groups.push([object]);
		// 		}
		// 	}
		// 	else if(object.type === "Group")
		// 	{
		// 		for (const child of object.children){
					
		// 		}
		// 	}
		// 	else 
		// 	{
		// 		console.error("unknown type of geometry! fallback to 0");
		// 		groups[0] = groups[0] || [];
		// 		groups[0].push(object);
		// 	}
		// }
		// this._groups = groups.filter(Array);
	},

	dispose: function () {

		this.renderTargetMaskBufferMain.dispose();
		for(const fbo of this.renderTargetMaskBuffer)
			fbo.dispose();

		this.renderTargetDepthBuffer.dispose();
		this.renderTargetMaskDownSampleBuffer.dispose();
		this.renderTargetBlurBuffer1.dispose();
		this.renderTargetBlurBuffer2.dispose();
		this.renderTargetEdgeBuffer1.dispose();
		this.renderTargetEdgeBuffer2.dispose();

	},

	setSize: function ( width, height ) {

		this.renderTargetMaskBufferMain.setSize( width, height );
		for(const fbo of this.renderTargetMaskBuffer)
			fbo.setSize( width, height );

		var resx = Math.round( width / this.downSampleRatio );
		var resy = Math.round( height / this.downSampleRatio );
		this.renderTargetMaskDownSampleBuffer.setSize( resx, resy );
		this.renderTargetBlurBuffer1.setSize( resx, resy );
		this.renderTargetEdgeBuffer1.setSize( resx, resy );
		this.separableBlurMaterial1.uniforms[ "texSize" ].value = new THREE.Vector2( resx, resy );

		resx = Math.round( resx / this.glowDownSampleRatio );
		resy = Math.round( resy / this.glowDownSampleRatio );

		this.renderTargetBlurBuffer2.setSize( resx, resy );
		this.renderTargetEdgeBuffer2.setSize( resx, resy );

		this.separableBlurMaterial2.uniforms[ "texSize" ].value = new THREE.Vector2( resx, resy );

	},

	changeVisibilityOfSelectedObjects: function ( bVisible, object ) {

		function gatherSelectedMeshesCallBack( object ) {

			if ( object.isMesh || object.isLine || object.isPoints ) {

				if ( bVisible ) {

					object.visible = object.userData.oldVisible;
					delete object.userData.oldVisible;

				} else {

					object.userData.oldVisible = object.visible;
					object.visible = bVisible;

				}

			}

		}

		for(const obj of (object || this._selectedObjects)){
			if(obj.geom){
				for(const geom of obj.geom)
					geom.traverse( gatherSelectedMeshesCallBack );
			} else {
				obj.traverse( gatherSelectedMeshesCallBack );
			}
		}

	},

	changeVisibilityOfNonSelectedObjects: function ( bVisible ) {

		var selectedMeshes = [];

		function gatherSelectedMeshesCallBack( object ) {

			if ( object.isMesh || object.isLine || object.isPoints ) selectedMeshes.push( object );

		}

		for(const obj of this._selectedObjects){
			for(const geom of obj.geom)
				geom.traverse( gatherSelectedMeshesCallBack );
		}

		function VisibilityChangeCallBack( object ) {

			if ( object.isMesh || object.isLine || object.isSprite || object.isPoints ) {

				var bFound = false;

				for ( var i = 0; i < selectedMeshes.length; i ++ ) {

					var selectedObjectId = selectedMeshes[ i ].id;

					if ( selectedObjectId === object.id ) {

						bFound = true;
						break;

					}

				}

				if ( ! bFound ) {

					var visibility = object.visible;

					if ( ! bVisible || object.bVisible ) object.visible = bVisible;

					object.bVisible = visibility;

				}

			}

		}

		this.renderScene.traverse( VisibilityChangeCallBack );

	},

	updateTextureMatrix: function () {

		this.textureMatrix.set( 0.5, 0.0, 0.0, 0.5,
			0.0, 0.5, 0.0, 0.5,
			0.0, 0.0, 0.5, 0.5,
			0.0, 0.0, 0.0, 1.0 );
		this.textureMatrix.multiply( this.renderCamera.projectionMatrix );
		this.textureMatrix.multiply( this.renderCamera.matrixWorldInverse );

	},

	draw: function(renderer, group, selection_type){	
		this.changeVisibilityOfSelectedObjects(true, group);	
		
		if(group[0].type === "Points"){
			this.prepareMaskMaterial.uniforms[ "pointSize" ].value = group[0].material.size || 20.0;
			if(group[0]["vertShader"]) this.prepareMaskMaterial.vertexShader = group[0]["vertShader"];
			if(group[0]["fragShader"]) this.prepareMaskMaterial.fragmentShader = group[0]["fragShader"];
		}
		renderer.setRenderTarget( this.renderTargetMaskBufferMain );
		renderer.render( this.renderScene, this.renderCamera );

		renderer.setRenderTarget( this.renderTargetMaskBuffer[selection_type] );
		renderer.render( this.renderScene, this.renderCamera );

		this.changeVisibilityOfSelectedObjects(false, group);	
	},

	render: function ( renderer, writeBuffer, readBuffer, deltaTime, maskActive ) {
		this._selectedObjects = Object.values(this.id2obj_map).flat();
		// console.log(this.selectedObjects);
		// debugger;

		if ( this._selectedObjects.length > 0 ) {
			// fetch objects that were created after secondary selection
			let sec_sel = this._selectedObjects.map(function(v){ 
				if(v.sec_sel) return v.geom;
			});
			sec_sel = sec_sel.flat();
			// debugger;

			// add sec_sel elements in renderScene
			for(const obj of sec_sel)
				if(obj) this.renderScene.add(obj);

			this.oldClearColor.copy( renderer.getClearColor() );
			this.oldClearAlpha = renderer.getClearAlpha();
			var oldAutoClear = renderer.autoClear;
			
			renderer.autoClear = false;

			if ( maskActive ) renderer.context.disable( renderer.context.STENCIL_TEST );
	
			renderer.setClearColor( 0xffffff, 1 );

			// Make selected objects invisible
			this.changeVisibilityOfSelectedObjects( false );
			
			var currentBackground = this.renderScene.background;
			this.renderScene.background = null;

			// 1. Draw Non Selected objects in the depth buffer
			this.renderScene.overrideMaterial = this.depthMaterial;
			renderer.setRenderTarget( this.renderTargetDepthBuffer );
			renderer.clear();
			renderer.render( this.renderScene, this.renderCamera );

			// Update Texture Matrix for Depth compare
			this.updateTextureMatrix();

			// Make non selected objects invisible, and draw only the selected objects, by comparing the depth buffer of non selected objects
			this.changeVisibilityOfNonSelectedObjects( false );

			this.renderScene.overrideMaterial = this.prepareMaskMaterial;
			this.prepareMaskMaterial.uniforms[ "cameraNearFar" ].value = new THREE.Vector2( this.renderCamera.near, this.renderCamera.far );
			this.prepareMaskMaterial.uniforms[ "depthTexture" ].value = this.renderTargetDepthBuffer.texture;
			this.prepareMaskMaterial.uniforms[ "textureMatrix" ].value = this.textureMatrix;

			renderer.setRenderTarget( this.renderTargetMaskBufferMain );
			renderer.clear();

			for(const maskBuffer of this.renderTargetMaskBuffer){
				renderer.setRenderTarget( maskBuffer );
				renderer.clear();
			}

			this.checkForCustomAtts();
			//console.log(this._groups);
			
			for(let i = 0; i < THREE.OutlinePass.selection_enum.total; ++i){
				for(const group of this._groups[i])
					if(group.length > 0)
						this.draw(renderer, group, i);
			}

			// if(this.atts.total > 0){
			// 	let group = [];

			// 	for(let i = 0; i < this.selectedObjects.length; ++i){
			// 		let p = this.atts["index"].indexOf(i);
			// 		if(p !== -1){
			// 			this.draw(renderer, writeBuffer, readBuffer, maskActive, this.selectedObjects[i], p);
			// 		} else {
			// 			group.push(this.selectedObjects[i]);
			// 		}
			// 	}

			// 	if(group.length > 0){

			// 	}
			// } else {
			// 	this.draw(renderer, writeBuffer, readBuffer, maskActive, this.selectedObjects, -1);
			// }

			this.renderScene.overrideMaterial = null;
			
			// enable all elements
			this.changeVisibilityOfSelectedObjects(true);
			this.changeVisibilityOfNonSelectedObjects(true);

			this.renderScene.background = currentBackground;
	
			// 2. Downsample to "downSampleRatio" res

			// clear stuff
			renderer.setRenderTarget( this.renderTargetEdgeBuffer1 );
			renderer.setClearColor( 0x000000, 0 )
			renderer.clear();
			renderer.setClearColor( 0xffffff, 1 );

			for(let i = 0; i < THREE.OutlinePass.selection_enum.total; ++i){
				let group = this._groups[i];
				if(group.length > 0){
					this.quad.material = this.materialCopy;
					this.copyUniforms[ "tDiffuse" ].value = this.renderTargetMaskBuffer[i].texture;
					renderer.setRenderTarget( this.renderTargetMaskDownSampleBuffer );
					renderer.clear();
					renderer.render( this.scene, this.camera );
	
					// 3. Apply Edge Detection Pass
					this.quad.material = this.edgeDetectionMaterial;
					this.edgeDetectionMaterial.uniforms[ "maskTexture" ].value = this.renderTargetMaskDownSampleBuffer.texture;
					this.edgeDetectionMaterial.uniforms[ "texSize" ].value = new THREE.Vector2( this.renderTargetMaskDownSampleBuffer.width, this.renderTargetMaskDownSampleBuffer.height );
	
					const att = THREE.OutlinePass.selection_atts[i];
					this.edgeDetectionMaterial.uniforms[ "visibleEdgeColor" ].value = att.visibleEdgeColor;
					this.edgeDetectionMaterial.uniforms[ "hiddenEdgeColor" ].value = att.hiddenEdgeColor;
					
					renderer.setRenderTarget( this.renderTargetEdgeBuffer1 );
					renderer.render( this.scene, this.camera );
				}
			}



			//>-----------------
			// for(let i = 0; i < THREE.OutlinePass.selection_enum.total; ++i){
			// 	const sel = this.sel[i];
			// 	if(sel.length > 0){
			// 		this.changeVisibilityOfSelectedObjects(true, sel);

			// 		// 3. Apply Edge Detection Pass
			// 		const att = THREE.OutlinePass.selection_atts[i];
			// 		this.edgeDetectionMaterial.uniforms[ "visibleEdgeColor" ].value = att.visibleEdgeColor;
			// 		this.edgeDetectionMaterial.uniforms[ "hiddenEdgeColor" ].value = att.hiddenEdgeColor;
			// 		renderer.render( this.scene, this.camera );

			// 		this.changeVisibilityOfSelectedObjects(false, sel);
			// 	}
			// }
			// this.changeVisibilityOfSelectedObjects(true);
	
			// 4. Apply Blur on "glowDownSampleRatio" res
			this.quad.material = this.separableBlurMaterial1;
			this.separableBlurMaterial1.uniforms[ "colorTexture" ].value = this.renderTargetEdgeBuffer1.texture;
			this.separableBlurMaterial1.uniforms[ "direction" ].value = THREE.OutlinePass.BlurDirectionX;
			this.separableBlurMaterial1.uniforms[ "kernelRadius" ].value = this.edgeThickness;
			renderer.setRenderTarget( this.renderTargetBlurBuffer1 );
			renderer.clear();
			renderer.render( this.scene, this.camera );
			this.separableBlurMaterial1.uniforms[ "colorTexture" ].value = this.renderTargetBlurBuffer1.texture;
			this.separableBlurMaterial1.uniforms[ "direction" ].value = THREE.OutlinePass.BlurDirectionY;
			renderer.setRenderTarget( this.renderTargetEdgeBuffer1 );
			renderer.clear();
			renderer.render( this.scene, this.camera );
	
			// Apply Blur on "downSampleRatio*glowDownSampleRatio" res
			this.quad.material = this.separableBlurMaterial2;
			this.separableBlurMaterial2.uniforms[ "colorTexture" ].value = this.renderTargetEdgeBuffer1.texture;
			this.separableBlurMaterial2.uniforms[ "direction" ].value = THREE.OutlinePass.BlurDirectionX;
			renderer.setRenderTarget( this.renderTargetBlurBuffer2 );
			renderer.clear();
			renderer.render( this.scene, this.camera );
			this.separableBlurMaterial2.uniforms[ "colorTexture" ].value = this.renderTargetBlurBuffer2.texture;
			this.separableBlurMaterial2.uniforms[ "direction" ].value = THREE.OutlinePass.BlurDirectionY;
			renderer.setRenderTarget( this.renderTargetEdgeBuffer2 );
			renderer.clear();
			renderer.render( this.scene, this.camera );
	
			// Blend it additively over the input texture
			this.quad.material = this.overlayMaterial;
			this.overlayMaterial.uniforms[ "maskTexture" ].value = this.renderTargetMaskBufferMain.texture;
			this.overlayMaterial.uniforms[ "edgeTexture1" ].value = this.renderTargetEdgeBuffer1.texture;
			this.overlayMaterial.uniforms[ "edgeTexture2" ].value = this.renderTargetEdgeBuffer2.texture;
			this.overlayMaterial.uniforms[ "patternTexture" ].value = this.patternTexture;
			this.overlayMaterial.uniforms[ "edgeStrength" ].value = this.edgeStrength;
			this.overlayMaterial.uniforms[ "edgeGlow" ].value = this.edgeGlow;
			this.overlayMaterial.uniforms[ "usePatternTexture" ].value = this.usePatternTexture;
	
			if ( maskActive ) renderer.context.enable( renderer.context.STENCIL_TEST );
	
			renderer.setRenderTarget( readBuffer );
			renderer.render( this.scene, this.camera );

			renderer.setClearColor( this.oldClearColor, this.oldClearAlpha );
			renderer.autoClear = oldAutoClear;
			
			// remove sec_sel elements from renderScene
			for(const obj of sec_sel)
				if(obj) this.renderScene.remove(obj);
		}
	},

	getPrepareMaskMaterial: function () {

		return new THREE.ShaderMaterial( {

			uniforms: {
				"depthTexture": { value: null },
				"cameraNearFar": { value: new THREE.Vector2( 0.5, 0.5 ) },
				"textureMatrix": { value: new THREE.Matrix4() },
				"pointSize": { value: 10.0 }
			},

			vertexShader:
				`varying vec4 projTexCoord;
				varying vec4 vPosition;
				uniform mat4 textureMatrix;
				uniform float pointSize;
				void main() {
					vPosition = modelViewMatrix * vec4( position, 1.0 );
					vec4 worldPosition = modelMatrix * vec4( position, 1.0 );
					projTexCoord = textureMatrix * worldPosition;
					gl_PointSize = pointSize;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,

			fragmentShader:
				`#include <packing>
				varying vec4 vPosition;
				varying vec4 projTexCoord;
				uniform sampler2D depthTexture;
				uniform vec2 cameraNearFar;
				void main() {
					float depth = unpackRGBAToDepth(texture2DProj( depthTexture, projTexCoord ));
					float viewZ = - DEPTH_TO_VIEW_Z( depth, cameraNearFar.x, cameraNearFar.y );
					float depthTest = (-vPosition.z > viewZ) ? 1.0 : 0.0;
					gl_FragColor = vec4(0.0, depthTest, 1.0, 1.0);
				}`
		} );
	},

	getEdgeDetectionMaterial: function () {

		return new THREE.ShaderMaterial( {

			uniforms: {
				"maskTexture": { value: null },
				"texSize": { value: new THREE.Vector2( 0.5, 0.5 ) },
				"visibleEdgeColor": { value: new THREE.Vector3( 1.0, 1.0, 1.0 ) },
				"hiddenEdgeColor": { value: new THREE.Vector3( 1.0, 1.0, 1.0 ) },
				"pointSize": { value: 10.0 }
			},

			vertexShader:
				"varying vec2 vUv;\n\
				uniform float pointSize;\n\
				void main() {\n\
					vUv = uv;\n\
					// gl_PointSize = pointSize;\n\
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\
				}",

			fragmentShader:
				`varying vec2 vUv;\
				uniform sampler2D maskTexture;\
				uniform vec2 texSize;\
				uniform vec3 visibleEdgeColor;\
				uniform vec3 hiddenEdgeColor;\
				\
				void main() {
				#if 1
					vec4 edge = texture2D( maskTexture, vUv);
					
					vec3 edgeColor = mix(visibleEdgeColor, hiddenEdgeColor, edge.g);
					gl_FragColor = vec4( edgeColor, 1.0 ) * (1.0-edge.r);
				#else
					vec2 invSize = 1.0 / texSize;
					vec4 uvOffset = vec4(1.0, 0.0, 0.0, 1.0) * vec4(invSize, invSize);
					vec4 c1 = texture2D( maskTexture, vUv + uvOffset.xy);
					vec4 c2 = texture2D( maskTexture, vUv - uvOffset.xy);
					vec4 c3 = texture2D( maskTexture, vUv + uvOffset.yw);
					vec4 c4 = texture2D( maskTexture, vUv - uvOffset.yw);
					float diff1 = (c1.r - c2.r)*0.5;
					float diff2 = (c3.r - c4.r)*0.5;
					float d = length( vec2(diff1, diff2) );
					float a1 = min(c1.g, c2.g);
					float a2 = min(c3.g, c4.g);
					float visibilityFactor = min(a1, a2);
					vec3 edgeColor = 1.0 - visibilityFactor > 0.001 ? visibleEdgeColor : hiddenEdgeColor;
					gl_FragColor = vec4(edgeColor, 1.0) * vec4(d);
				#endif
				}`,
				blending: THREE.CustomBlending,
				blendEquation: THREE.AddEquation,
				blendSrc: THREE.OneMinusDstAlphaFactor,
				blendDst: THREE.DstAlphaFactor,
				depthTest: false,
				depthWrite: false,
				transparent: false
		} );

	},

	getSeperableBlurMaterial: function ( maxRadius ) {

		return new THREE.ShaderMaterial( {

			defines: {
				"MAX_RADIUS": maxRadius,
			},

			uniforms: {
				"colorTexture": { value: null },
				"texSize": { value: new THREE.Vector2( 0.5, 0.5 ) },
				"direction": { value: new THREE.Vector2( 0.5, 0.5 ) },
				"kernelRadius": { value: 1.0 },
				"pointSize": { value: 10.0 }
			},

			vertexShader:
				"varying vec2 vUv;\n\
				uniform float pointSize;\n\
				void main() {\n\
					vUv = uv;\n\
					// gl_PointSize = pointSize;\n\
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n\
				}",

			fragmentShader:
				"#include <common>\
				varying vec2 vUv;\
				uniform sampler2D colorTexture;\
				uniform vec2 texSize;\
				uniform vec2 direction;\
				uniform float kernelRadius;\
				\
				float gaussianPdf(in float x, in float sigma) {\
					return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;\
				}\
				void main() {\
					vec2 invSize = 1.0 / texSize;\
					float weightSum = gaussianPdf(0.0, kernelRadius);\
					vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;\
					vec2 delta = direction * invSize * kernelRadius/float(MAX_RADIUS);\
					vec2 uvOffset = delta;\
					for( int i = 1; i <= MAX_RADIUS; i ++ ) {\
						float w = gaussianPdf(uvOffset.x, kernelRadius);\
						vec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;\
						vec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;\
						diffuseSum += ((sample1 + sample2) * w);\
						weightSum += (2.0 * w);\
						uvOffset += delta;\
					}\
					gl_FragColor = vec4(diffuseSum/weightSum, 1.0);\
				}"
		} );

	},

	getOverlayMaterial: function () {

		return new THREE.ShaderMaterial( {

			uniforms: {
				"maskTexture": { value: null },
				"edgeTexture1": { value: null },
				"edgeTexture2": { value: null },
				"patternTexture": { value: null },
				"edgeStrength": { value: 1.0 },
				"edgeGlow": { value: 1.0 },
				"usePatternTexture": { value: 0.0 },
				"pointSize": { value: 10.0 }
			},

			vertexShader:
				`varying vec2 vUv;
				uniform float pointSize;
				void main() {
					vUv = uv;
					// gl_PointSize = pointSize;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,

			fragmentShader:
				`varying vec2 vUv;
				precision highp float;
				uniform sampler2D maskTexture;
				uniform sampler2D edgeTexture1;
				uniform sampler2D edgeTexture2;
				uniform sampler2D patternTexture;
				uniform float edgeStrength;
				uniform float edgeGlow;
				uniform bool usePatternTexture;

				void main() {
					vec3 edgeValue1 = texture2D(edgeTexture1, vUv).rgb;
					vec3 edgeValue2 = texture2D(edgeTexture2, vUv).rgb;
					vec4 maskColor = texture2D(maskTexture, vUv);
					float visibilityFactor = 1.0 - maskColor.g > 0.0 ? 1.0 : 0.5;
					vec3 edgeValue = edgeValue1 + edgeValue2 * edgeGlow;
					vec3 finalColor = edgeStrength * edgeValue;
				#if 0
					vec4 patternColor = texture2D(patternTexture, 6.0 * vUv);
					if(usePatternTexture)
						finalColor += + visibilityFactor * (1.0 - maskColor.r) * (1.0 - patternColor.r);
				#endif
					gl_FragColor = vec4(finalColor, maskColor.r * dot(finalColor.rgb, vec3(0.33333)));
				}`,
				blending: THREE.CustomBlending,
				blendEquation: THREE.AddEquation,
				blendSrc: THREE.SrcAlphaFactor,
				blendDst: THREE.OneMinusSrcAlphaFactor,
				depthTest: false,
				depthWrite: false,
				transparent: false
		} );

	}

} );

if(false){
	THREE.OutlinePass.customAtt = function(mesh, opts){
		opts = opts || {};
	
		if(opts.index === undefined)
			throw "you gotta pass a valid index";
		this.index = opts.index;
	
		this.isPoint = mesh.isPoints || opts.isPoints || false;
		if(this.isPoint)
			this.pointSize = opts.pointSize || 20;
	
		this.hasCustomVertShader = (opts.vertShader !== undefined);
		if(this.hasCustomVertShader) this.vertShader = opts.vertShader;
	
		this.hasCustomFragShader = (opts.fragShader !== undefined);
		if(this.hasCustomFragShader) this.fragShader = opts.fragShader;
	};
}

THREE.OutlinePass.BlurDirectionX = new THREE.Vector2( 1.0, 0.0 );
THREE.OutlinePass.BlurDirectionY = new THREE.Vector2( 0.0, 1.0 );

THREE.OutlinePass.selection_enum = {
	"select": 0,
	"highlight": 1,
	"total": 2 
};

THREE.OutlinePass.selection_atts = [
	{
		visibleEdgeColor: new THREE.Color( 1, 0, 0 ),
		hiddenEdgeColor: new THREE.Color( 1, 0, 0.5 )
		// hiddenEdgeColoe: new THREE.Color( 0.1, 0.04, 0.02 )
		// edgeGlow: 0.0,
		// usePatternTexture: false,
		// edgeThickness: 1.0,
		// edgeStrength: 3.0,
	},
	{
		visibleEdgeColor: new THREE.Color( 0, 1, 0 ),
		hiddenEdgeColor: new THREE.Color( 0, 1, 0.5 )
		// hiddenEdgeColor: new THREE.Color( 0.1, 0.04, 0.02 )
		// edgeGlow: 0.0,
		// usePatternTexture: false,
		// edgeThickness: 1.0,
		// edgeStrength: 3.0,
	}
];
