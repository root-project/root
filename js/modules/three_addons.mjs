/**
 * @license
 * Copyright 2010-2023 Three.js Authors
 * SPDX-License-Identifier: MIT
 */
import { ExtrudeGeometry, ShapePath, Ray, Plane, MathUtils, EventDispatcher, Vector3, MOUSE, TOUCH, Quaternion, Spherical, Vector2, OrthographicCamera, BufferGeometry, Float32BufferAttribute, Mesh, ShaderMaterial, UniformsUtils, WebGLRenderTarget, HalfFloatType, NoBlending, Clock, Color, AdditiveBlending, MeshBasicMaterial, Vector4, Box3, Matrix4, Frustum, Matrix3, DoubleSide, Box2, SRGBColorSpace, Camera } from './three.mjs';

/**
 * Text = 3D Text
 *
 * parameters = {
 *  font: <THREE.Font>, // font
 *
 *  size: <float>, // size of the text
 *  height: <float>, // thickness to extrude text
 *  curveSegments: <int>, // number of points on the curves
 *
 *  bevelEnabled: <bool>, // turn on bevel
 *  bevelThickness: <float>, // how deep into text bevel goes
 *  bevelSize: <float>, // how far from text outline (including bevelOffset) is bevel
 *  bevelOffset: <float> // how far from text outline does bevel start
 * }
 */


class TextGeometry extends ExtrudeGeometry {

	constructor( text, parameters = {} ) {

		const font = parameters.font;

		if ( font === undefined ) {

			super(); // generate default extrude geometry

		} else {

			const shapes = font.generateShapes( text, parameters.size );

			// translate parameters to ExtrudeGeometry API

			parameters.depth = parameters.height !== undefined ? parameters.height : 50;

			// defaults

			if ( parameters.bevelThickness === undefined ) parameters.bevelThickness = 10;
			if ( parameters.bevelSize === undefined ) parameters.bevelSize = 8;
			if ( parameters.bevelEnabled === undefined ) parameters.bevelEnabled = false;

			super( shapes, parameters );

		}

		this.type = 'TextGeometry';

	}

}

//

class Font {

	constructor( data ) {

		this.isFont = true;

		this.type = 'Font';

		this.data = data;

	}

	generateShapes( text, size = 100 ) {

		const shapes = [];
		const paths = createPaths( text, size, this.data );

		for ( let p = 0, pl = paths.length; p < pl; p ++ ) {

			shapes.push( ...paths[ p ].toShapes() );

		}

		return shapes;

	}

}

function createPaths( text, size, data ) {

	const chars = Array.from( text );
	const scale = size / data.resolution;
	const line_height = ( data.boundingBox.yMax - data.boundingBox.yMin + data.underlineThickness ) * scale;

	const paths = [];

	let offsetX = 0, offsetY = 0;

	for ( let i = 0; i < chars.length; i ++ ) {

		const char = chars[ i ];

		if ( char === '\n' ) {

			offsetX = 0;
			offsetY -= line_height;

		} else {

			const ret = createPath( char, scale, offsetX, offsetY, data );
			offsetX += ret.offsetX;
			paths.push( ret.path );

		}

	}

	return paths;

}

function createPath( char, scale, offsetX, offsetY, data ) {

	const glyph = data.glyphs[ char ] || data.glyphs[ '?' ];

	if ( ! glyph ) {

		console.error( 'THREE.Font: character "' + char + '" does not exists in font family ' + data.familyName + '.' );

		return;

	}

	const path = new ShapePath();

	let x, y, cpx, cpy, cpx1, cpy1, cpx2, cpy2;

	if ( glyph.o ) {

		const outline = glyph._cachedOutline || ( glyph._cachedOutline = glyph.o.split( ' ' ) );

		for ( let i = 0, l = outline.length; i < l; ) {

			const action = outline[ i ++ ];

			switch ( action ) {

				case 'm': // moveTo

					x = outline[ i ++ ] * scale + offsetX;
					y = outline[ i ++ ] * scale + offsetY;

					path.moveTo( x, y );

					break;

				case 'l': // lineTo

					x = outline[ i ++ ] * scale + offsetX;
					y = outline[ i ++ ] * scale + offsetY;

					path.lineTo( x, y );

					break;

				case 'q': // quadraticCurveTo

					cpx = outline[ i ++ ] * scale + offsetX;
					cpy = outline[ i ++ ] * scale + offsetY;
					cpx1 = outline[ i ++ ] * scale + offsetX;
					cpy1 = outline[ i ++ ] * scale + offsetY;

					path.quadraticCurveTo( cpx1, cpy1, cpx, cpy );

					break;

				case 'b': // bezierCurveTo

					cpx = outline[ i ++ ] * scale + offsetX;
					cpy = outline[ i ++ ] * scale + offsetY;
					cpx1 = outline[ i ++ ] * scale + offsetX;
					cpy1 = outline[ i ++ ] * scale + offsetY;
					cpx2 = outline[ i ++ ] * scale + offsetX;
					cpy2 = outline[ i ++ ] * scale + offsetY;

					path.bezierCurveTo( cpx1, cpy1, cpx2, cpy2, cpx, cpy );

					break;

			}

		}

	}

	return { offsetX: glyph.ha * scale, path: path };

}

// OrbitControls performs orbiting, dollying (zooming), and panning.
// Unlike TrackballControls, it maintains the "up" direction object.up (+Y by default).
//
//    Orbit - left mouse / touch: one-finger move
//    Zoom - middle mouse, or mousewheel / touch: two-finger spread or squish
//    Pan - right mouse, or left mouse + ctrl/meta/shiftKey, or arrow keys / touch: two-finger move

const _changeEvent = { type: 'change' };
const _startEvent = { type: 'start' };
const _endEvent = { type: 'end' };
const _ray = new Ray();
const _plane = new Plane();
const TILT_LIMIT = Math.cos( 70 * MathUtils.DEG2RAD );

class OrbitControls extends EventDispatcher {

	constructor( object, domElement ) {

		super();

		this.object = object;
		this.domElement = domElement;
		this.domElement.style.touchAction = 'none'; // disable touch scroll

		// Set to false to disable this control
		this.enabled = true;

		// "target" sets the location of focus, where the object orbits around
		this.target = new Vector3();

		// Sets the 3D cursor (similar to Blender), from which the maxTargetRadius takes effect
		this.cursor = new Vector3();

		// How far you can dolly in and out ( PerspectiveCamera only )
		this.minDistance = 0;
		this.maxDistance = Infinity;

		// How far you can zoom in and out ( OrthographicCamera only )
		this.minZoom = 0;
		this.maxZoom = Infinity;

		// Limit camera target within a spherical area around the cursor
		this.minTargetRadius = 0;
		this.maxTargetRadius = Infinity;

		// How far you can orbit vertically, upper and lower limits.
		// Range is 0 to Math.PI radians.
		this.minPolarAngle = 0; // radians
		this.maxPolarAngle = Math.PI; // radians

		// How far you can orbit horizontally, upper and lower limits.
		// If set, the interval [ min, max ] must be a sub-interval of [ - 2 PI, 2 PI ], with ( max - min < 2 PI )
		this.minAzimuthAngle = - Infinity; // radians
		this.maxAzimuthAngle = Infinity; // radians

		// Set to true to enable damping (inertia)
		// If damping is enabled, you must call controls.update() in your animation loop
		this.enableDamping = false;
		this.dampingFactor = 0.05;

		// This option actually enables dollying in and out; left as "zoom" for backwards compatibility.
		// Set to false to disable zooming
		this.enableZoom = true;
		this.zoomSpeed = 1.0;

		// Set to false to disable rotating
		this.enableRotate = true;
		this.rotateSpeed = 1.0;

		// Set to false to disable panning
		this.enablePan = true;
		this.panSpeed = 1.0;
		this.screenSpacePanning = true; // if false, pan orthogonal to world-space direction camera.up
		this.keyPanSpeed = 7.0;	// pixels moved per arrow key push
		this.zoomToCursor = false;

		// Set to true to automatically rotate around the target
		// If auto-rotate is enabled, you must call controls.update() in your animation loop
		this.autoRotate = false;
		this.autoRotateSpeed = 2.0; // 30 seconds per orbit when fps is 60

		// The four arrow keys
		this.keys = { LEFT: 'ArrowLeft', UP: 'ArrowUp', RIGHT: 'ArrowRight', BOTTOM: 'ArrowDown' };

		// Mouse buttons
		this.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };

		// Touch fingers
		this.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };

		// for reset
		this.target0 = this.target.clone();
		this.position0 = this.object.position.clone();
		this.zoom0 = this.object.zoom;

		// the target DOM element for key events
		this._domElementKeyEvents = null;

		//
		// public methods
		//

		this.getPolarAngle = function () {

			return spherical.phi;

		};

		this.getAzimuthalAngle = function () {

			return spherical.theta;

		};

		this.getDistance = function () {

			return this.object.position.distanceTo( this.target );

		};

		this.listenToKeyEvents = function ( domElement ) {

			domElement.addEventListener( 'keydown', onKeyDown );
			this._domElementKeyEvents = domElement;

		};

		this.stopListenToKeyEvents = function () {

			this._domElementKeyEvents.removeEventListener( 'keydown', onKeyDown );
			this._domElementKeyEvents = null;

		};

		this.saveState = function () {

			scope.target0.copy( scope.target );
			scope.position0.copy( scope.object.position );
			scope.zoom0 = scope.object.zoom;

		};

		this.reset = function () {

			scope.target.copy( scope.target0 );
			scope.object.position.copy( scope.position0 );
			scope.object.zoom = scope.zoom0;

			scope.object.updateProjectionMatrix();
			scope.dispatchEvent( _changeEvent );

			scope.update();

			state = STATE.NONE;

		};

		this.resetOrthoPanZoom = function () {
         panOffset.set(0,0,0);
         scope.object.zoom = 1;
         scope.object.updateProjectionMatrix();
      };

		// this method is exposed, but perhaps it would be better if we can make it private...
		this.update = function () {

			const offset = new Vector3();

			// so camera.up is the orbit axis
			const quat = new Quaternion().setFromUnitVectors( object.up, new Vector3( 0, 1, 0 ) );
			const quatInverse = quat.clone().invert();

			const lastPosition = new Vector3();
			const lastQuaternion = new Quaternion();
			const lastTargetPosition = new Vector3();

			const twoPI = 2 * Math.PI;

			return function update( deltaTime = null ) {

				const position = scope.object.position;

				offset.copy( position ).sub( scope.target );

				// rotate offset to "y-axis-is-up" space
				offset.applyQuaternion( quat );

				// angle from z-axis around y-axis
				spherical.setFromVector3( offset );

				if ( scope.autoRotate && state === STATE.NONE ) {

					rotateLeft( getAutoRotationAngle( deltaTime ) );

				}

				if ( scope.enableDamping ) {

					spherical.theta += sphericalDelta.theta * scope.dampingFactor;
					spherical.phi += sphericalDelta.phi * scope.dampingFactor;

				} else {

					spherical.theta += sphericalDelta.theta;
					spherical.phi += sphericalDelta.phi;

				}

				// restrict theta to be between desired limits

				let min = scope.minAzimuthAngle;
				let max = scope.maxAzimuthAngle;

				if ( isFinite( min ) && isFinite( max ) ) {

					if ( min < - Math.PI ) min += twoPI; else if ( min > Math.PI ) min -= twoPI;

					if ( max < - Math.PI ) max += twoPI; else if ( max > Math.PI ) max -= twoPI;

					if ( min <= max ) {

						spherical.theta = Math.max( min, Math.min( max, spherical.theta ) );

					} else {

						spherical.theta = ( spherical.theta > ( min + max ) / 2 ) ?
							Math.max( min, spherical.theta ) :
							Math.min( max, spherical.theta );

					}

				}

				// restrict phi to be between desired limits
				spherical.phi = Math.max( scope.minPolarAngle, Math.min( scope.maxPolarAngle, spherical.phi ) );

				spherical.makeSafe();


				// move target to panned location

				if ( scope.enableDamping === true ) {

					scope.target.addScaledVector( panOffset, scope.dampingFactor );

				} else {

					scope.target.add( panOffset );

				}

				// Limit the target distance from the cursor to create a sphere around the center of interest
				scope.target.sub( scope.cursor );
				scope.target.clampLength( scope.minTargetRadius, scope.maxTargetRadius );
				scope.target.add( scope.cursor );

				let zoomChanged = false;
				// adjust the camera position based on zoom only if we're not zooming to the cursor or if it's an ortho camera
				// we adjust zoom later in these cases
				if ( scope.zoomToCursor && performCursorZoom || scope.object.isOrthographicCamera ) {

					spherical.radius = clampDistance( spherical.radius );

				} else {

					const prevRadius = spherical.radius;
					spherical.radius = clampDistance( spherical.radius * scale );
					zoomChanged = prevRadius != spherical.radius;

				}

				offset.setFromSpherical( spherical );

				// rotate offset back to "camera-up-vector-is-up" space
				offset.applyQuaternion( quatInverse );

				position.copy( scope.target ).add( offset );

				scope.object.lookAt( scope.target );

				if ( scope.enableDamping === true ) {

					sphericalDelta.theta *= ( 1 - scope.dampingFactor );
					sphericalDelta.phi *= ( 1 - scope.dampingFactor );

					panOffset.multiplyScalar( 1 - scope.dampingFactor );

				} else {

					sphericalDelta.set( 0, 0, 0 );

					panOffset.set( 0, 0, 0 );

				}

				// adjust camera position
				if ( scope.zoomToCursor && performCursorZoom ) {

					let newRadius = null;
					if ( scope.object.isPerspectiveCamera ) {

						// move the camera down the pointer ray
						// this method avoids floating point error
						const prevRadius = offset.length();
						newRadius = clampDistance( prevRadius * scale );

						const radiusDelta = prevRadius - newRadius;
						scope.object.position.addScaledVector( dollyDirection, radiusDelta );
						scope.object.updateMatrixWorld();

						zoomChanged = !! radiusDelta;

					} else if ( scope.object.isOrthographicCamera ) {

						// adjust the ortho camera position based on zoom changes
						const mouseBefore = new Vector3( mouse.x, mouse.y, 0 );
						mouseBefore.unproject( scope.object );

						const prevZoom = scope.object.zoom;
						scope.object.zoom = Math.max( scope.minZoom, Math.min( scope.maxZoom, scope.object.zoom / scale ) );
						scope.object.updateProjectionMatrix();

						zoomChanged = prevZoom !== scope.object.zoom;

						const mouseAfter = new Vector3( mouse.x, mouse.y, 0 );
						mouseAfter.unproject( scope.object );

						scope.object.position.sub( mouseAfter ).add( mouseBefore );
						scope.object.updateMatrixWorld();

						newRadius = offset.length();

					} else {

						console.warn( 'WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled.' );
						scope.zoomToCursor = false;

					}

					// handle the placement of the target
					if ( newRadius !== null ) {

						if ( this.screenSpacePanning ) {

							// position the orbit target in front of the new camera position
							scope.target.set( 0, 0, - 1 )
								.transformDirection( scope.object.matrix )
								.multiplyScalar( newRadius )
								.add( scope.object.position );

						} else {

							// get the ray and translation plane to compute target
							_ray.origin.copy( scope.object.position );
							_ray.direction.set( 0, 0, - 1 ).transformDirection( scope.object.matrix );

							// if the camera is 20 degrees above the horizon then don't adjust the focus target to avoid
							// extremely large values
							if ( Math.abs( scope.object.up.dot( _ray.direction ) ) < TILT_LIMIT ) {

								object.lookAt( scope.target );

							} else {

								_plane.setFromNormalAndCoplanarPoint( scope.object.up, scope.target );
								_ray.intersectPlane( _plane, scope.target );

							}

						}

					}

				} else if ( scope.object.isOrthographicCamera ) {

					const prevZoom = scope.object.zoom;
					scope.object.zoom = Math.max( scope.minZoom, Math.min( scope.maxZoom, scope.object.zoom / scale ) );

					if ( prevZoom !== scope.object.zoom ) {

						scope.object.updateProjectionMatrix();
						zoomChanged = true;

					}

				}

				scale = 1;
				performCursorZoom = false;

				// update condition is:
				// min(camera displacement, camera rotation in radians)^2 > EPS
				// using small-angle approximation cos(x/2) = 1 - x^2 / 8

				if ( zoomChanged ||
					lastPosition.distanceToSquared( scope.object.position ) > EPS ||
					8 * ( 1 - lastQuaternion.dot( scope.object.quaternion ) ) > EPS ||
					lastTargetPosition.distanceToSquared( scope.target ) > EPS ) {

					scope.dispatchEvent( _changeEvent );

					lastPosition.copy( scope.object.position );
					lastQuaternion.copy( scope.object.quaternion );
					lastTargetPosition.copy( scope.target );

					return true;

				}

				return false;

			};

		}();

		this.dispose = function () {

			scope.domElement.removeEventListener( 'contextmenu', onContextMenu );

			scope.domElement.removeEventListener( 'pointerdown', onPointerDown );
			scope.domElement.removeEventListener( 'pointercancel', onPointerUp );
			scope.domElement.removeEventListener( 'wheel', onMouseWheel );

			scope.domElement.removeEventListener( 'pointermove', onPointerMove );
			scope.domElement.removeEventListener( 'pointerup', onPointerUp );

			const document = scope.domElement.getRootNode(); // offscreen canvas compatibility

			document.removeEventListener( 'keydown', interceptControlDown, { capture: true } );

			if ( scope._domElementKeyEvents !== null ) {

				scope._domElementKeyEvents.removeEventListener( 'keydown', onKeyDown );
				scope._domElementKeyEvents = null;

			}

			//scope.dispatchEvent( { type: 'dispose' } ); // should this be added here?

		};

		//
		// internals
		//

		const scope = this;

		const STATE = {
			NONE: - 1,
			ROTATE: 0,
			DOLLY: 1,
			PAN: 2,
			TOUCH_ROTATE: 3,
			TOUCH_PAN: 4,
			TOUCH_DOLLY_PAN: 5,
			TOUCH_DOLLY_ROTATE: 6
		};

		let state = STATE.NONE;

		const EPS = 0.000001;

		// current position in spherical coordinates
		const spherical = new Spherical();
		const sphericalDelta = new Spherical();

		let scale = 1;
		const panOffset = new Vector3();

		const rotateStart = new Vector2();
		const rotateEnd = new Vector2();
		const rotateDelta = new Vector2();

		const panStart = new Vector2();
		const panEnd = new Vector2();
		const panDelta = new Vector2();

		const dollyStart = new Vector2();
		const dollyEnd = new Vector2();
		const dollyDelta = new Vector2();

		const dollyDirection = new Vector3();
		const mouse = new Vector2();
		let performCursorZoom = false;

		const pointers = [];
		const pointerPositions = {};

		let controlActive = false;

		function getAutoRotationAngle( deltaTime ) {

			if ( deltaTime !== null ) {

				return ( 2 * Math.PI / 60 * scope.autoRotateSpeed ) * deltaTime;

			} else {

				return 2 * Math.PI / 60 / 60 * scope.autoRotateSpeed;

			}

		}

		function getZoomScale( delta ) {

			const normalizedDelta = Math.abs( delta * 0.01 );
			return Math.pow( 0.95, scope.zoomSpeed * normalizedDelta );

		}

		function rotateLeft( angle ) {

			sphericalDelta.theta -= angle;

		}

		function rotateUp( angle ) {

			sphericalDelta.phi -= angle;

		}

		const panLeft = function () {

			const v = new Vector3();

			return function panLeft( distance, objectMatrix ) {

				v.setFromMatrixColumn( objectMatrix, 0 ); // get X column of objectMatrix
				v.multiplyScalar( - distance );

				panOffset.add( v );

			};

		}();

		const panUp = function () {

			const v = new Vector3();

			return function panUp( distance, objectMatrix ) {

				if ( scope.screenSpacePanning === true ) {

					v.setFromMatrixColumn( objectMatrix, 1 );

				} else {

					v.setFromMatrixColumn( objectMatrix, 0 );
					v.crossVectors( scope.object.up, v );

				}

				v.multiplyScalar( distance );

				panOffset.add( v );

			};

		}();

		// deltaX and deltaY are in pixels; right and down are positive
		const pan = function () {

			const offset = new Vector3();

			return function pan( deltaX, deltaY ) {

				const element = scope.domElement;

				if ( scope.object.isPerspectiveCamera ) {

					// perspective
					const position = scope.object.position;
					offset.copy( position ).sub( scope.target );
					let targetDistance = offset.length();

					// half of the fov is center to top of screen
					targetDistance *= Math.tan( ( scope.object.fov / 2 ) * Math.PI / 180.0 );

					// we use only clientHeight here so aspect ratio does not distort speed
					panLeft( 2 * deltaX * targetDistance / element.clientHeight, scope.object.matrix );
					panUp( 2 * deltaY * targetDistance / element.clientHeight, scope.object.matrix );

				} else if ( scope.object.isOrthographicCamera ) {

					// orthographic
					panLeft( deltaX * ( scope.object.right - scope.object.left ) / scope.object.zoom / element.clientWidth, scope.object.matrix );
					panUp( deltaY * ( scope.object.top - scope.object.bottom ) / scope.object.zoom / element.clientHeight, scope.object.matrix );

				} else {

					// camera neither orthographic nor perspective
					console.warn( 'WARNING: OrbitControls.js encountered an unknown camera type - pan disabled.' );
					scope.enablePan = false;

				}

			};

		}();

		function dollyOut( dollyScale ) {

			if ( scope.object.isPerspectiveCamera || scope.object.isOrthographicCamera ) {

				scale /= dollyScale;

			} else {

				console.warn( 'WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled.' );
				scope.enableZoom = false;

			}

		}

		function dollyIn( dollyScale ) {

			if ( scope.object.isPerspectiveCamera || scope.object.isOrthographicCamera ) {

				scale *= dollyScale;

			} else {

				console.warn( 'WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled.' );
				scope.enableZoom = false;

			}

		}

		function updateZoomParameters( x, y ) {

			if ( ! scope.zoomToCursor ) {

				return;

			}

			performCursorZoom = true;

			const rect = scope.domElement.getBoundingClientRect();
			const dx = x - rect.left;
			const dy = y - rect.top;
			const w = rect.width;
			const h = rect.height;

			mouse.x = ( dx / w ) * 2 - 1;
			mouse.y = - ( dy / h ) * 2 + 1;

			dollyDirection.set( mouse.x, mouse.y, 1 ).unproject( scope.object ).sub( scope.object.position ).normalize();

		}

		function clampDistance( dist ) {

			return Math.max( scope.minDistance, Math.min( scope.maxDistance, dist ) );

		}

		//
		// event callbacks - update the object state
		//

		function handleMouseDownRotate( event ) {

			rotateStart.set( event.clientX, event.clientY );

		}

		function handleMouseDownDolly( event ) {

			updateZoomParameters( event.clientX, event.clientX );
			dollyStart.set( event.clientX, event.clientY );

		}

		function handleMouseDownPan( event ) {

			panStart.set( event.clientX, event.clientY );

		}

		function handleMouseMoveRotate( event ) {

			rotateEnd.set( event.clientX, event.clientY );

			rotateDelta.subVectors( rotateEnd, rotateStart ).multiplyScalar( scope.rotateSpeed );

			const element = scope.domElement;

			rotateLeft( 2 * Math.PI * rotateDelta.x / element.clientHeight ); // yes, height

			rotateUp( 2 * Math.PI * rotateDelta.y / element.clientHeight );

			rotateStart.copy( rotateEnd );

			scope.update();

		}

		function handleMouseMoveDolly( event ) {

			dollyEnd.set( event.clientX, event.clientY );

			dollyDelta.subVectors( dollyEnd, dollyStart );

			if ( dollyDelta.y > 0 ) {

				dollyOut( getZoomScale( dollyDelta.y ) );

			} else if ( dollyDelta.y < 0 ) {

				dollyIn( getZoomScale( dollyDelta.y ) );

			}

			dollyStart.copy( dollyEnd );

			scope.update();

		}

		function handleMouseMovePan( event ) {

			panEnd.set( event.clientX, event.clientY );

			panDelta.subVectors( panEnd, panStart ).multiplyScalar( scope.panSpeed );

			pan( panDelta.x, panDelta.y );

			panStart.copy( panEnd );

			scope.update();

		}

		function handleMouseWheel( event ) {

			updateZoomParameters( event.clientX, event.clientY );

			if ( event.deltaY < 0 ) {

				dollyIn( getZoomScale( event.deltaY ) );

			} else if ( event.deltaY > 0 ) {

				dollyOut( getZoomScale( event.deltaY ) );

			}

			scope.update();

		}

		function handleKeyDown( event ) {

			let needsUpdate = false;

			switch ( event.code ) {

				case scope.keys.UP:

					if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

						rotateUp( 2 * Math.PI * scope.rotateSpeed / scope.domElement.clientHeight );

					} else {

						pan( 0, scope.keyPanSpeed );

					}

					needsUpdate = true;
					break;

				case scope.keys.BOTTOM:

					if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

						rotateUp( - 2 * Math.PI * scope.rotateSpeed / scope.domElement.clientHeight );

					} else {

						pan( 0, - scope.keyPanSpeed );

					}

					needsUpdate = true;
					break;

				case scope.keys.LEFT:

					if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

						rotateLeft( 2 * Math.PI * scope.rotateSpeed / scope.domElement.clientHeight );

					} else {

						pan( scope.keyPanSpeed, 0 );

					}

					needsUpdate = true;
					break;

				case scope.keys.RIGHT:

					if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

						rotateLeft( - 2 * Math.PI * scope.rotateSpeed / scope.domElement.clientHeight );

					} else {

						pan( - scope.keyPanSpeed, 0 );

					}

					needsUpdate = true;
					break;

			}

			if ( needsUpdate ) {

				// prevent the browser from scrolling on cursor keys
				event.preventDefault();

				scope.update();

			}


		}

		function handleTouchStartRotate( event ) {

			if ( pointers.length === 1 ) {

				rotateStart.set( event.pageX, event.pageY );

			} else {

				const position = getSecondPointerPosition( event );

				const x = 0.5 * ( event.pageX + position.x );
				const y = 0.5 * ( event.pageY + position.y );

				rotateStart.set( x, y );

			}

		}

		function handleTouchStartPan( event ) {

			if ( pointers.length === 1 ) {

				panStart.set( event.pageX, event.pageY );

			} else {

				const position = getSecondPointerPosition( event );

				const x = 0.5 * ( event.pageX + position.x );
				const y = 0.5 * ( event.pageY + position.y );

				panStart.set( x, y );

			}

		}

		function handleTouchStartDolly( event ) {

			const position = getSecondPointerPosition( event );

			const dx = event.pageX - position.x;
			const dy = event.pageY - position.y;

			const distance = Math.sqrt( dx * dx + dy * dy );

			dollyStart.set( 0, distance );

		}

		function handleTouchStartDollyPan( event ) {

			if ( scope.enableZoom ) handleTouchStartDolly( event );

			if ( scope.enablePan ) handleTouchStartPan( event );

		}

		function handleTouchStartDollyRotate( event ) {

			if ( scope.enableZoom ) handleTouchStartDolly( event );

			if ( scope.enableRotate ) handleTouchStartRotate( event );

		}

		function handleTouchMoveRotate( event ) {

			if ( pointers.length == 1 ) {

				rotateEnd.set( event.pageX, event.pageY );

			} else {

				const position = getSecondPointerPosition( event );

				const x = 0.5 * ( event.pageX + position.x );
				const y = 0.5 * ( event.pageY + position.y );

				rotateEnd.set( x, y );

			}

			rotateDelta.subVectors( rotateEnd, rotateStart ).multiplyScalar( scope.rotateSpeed );

			const element = scope.domElement;

			rotateLeft( 2 * Math.PI * rotateDelta.x / element.clientHeight ); // yes, height

			rotateUp( 2 * Math.PI * rotateDelta.y / element.clientHeight );

			rotateStart.copy( rotateEnd );

		}

		function handleTouchMovePan( event ) {

			if ( pointers.length === 1 ) {

				panEnd.set( event.pageX, event.pageY );

			} else {

				const position = getSecondPointerPosition( event );

				const x = 0.5 * ( event.pageX + position.x );
				const y = 0.5 * ( event.pageY + position.y );

				panEnd.set( x, y );

			}

			panDelta.subVectors( panEnd, panStart ).multiplyScalar( scope.panSpeed );

			pan( panDelta.x, panDelta.y );

			panStart.copy( panEnd );

		}

		function handleTouchMoveDolly( event ) {

			const position = getSecondPointerPosition( event );

			const dx = event.pageX - position.x;
			const dy = event.pageY - position.y;

			const distance = Math.sqrt( dx * dx + dy * dy );

			dollyEnd.set( 0, distance );

			dollyDelta.set( 0, Math.pow( dollyEnd.y / dollyStart.y, scope.zoomSpeed ) );

			dollyOut( dollyDelta.y );

			dollyStart.copy( dollyEnd );

			const centerX = ( event.pageX + position.x ) * 0.5;
			const centerY = ( event.pageY + position.y ) * 0.5;

			updateZoomParameters( centerX, centerY );

		}

		function handleTouchMoveDollyPan( event ) {

			if ( scope.enableZoom ) handleTouchMoveDolly( event );

			if ( scope.enablePan ) handleTouchMovePan( event );

		}

		function handleTouchMoveDollyRotate( event ) {

			if ( scope.enableZoom ) handleTouchMoveDolly( event );

			if ( scope.enableRotate ) handleTouchMoveRotate( event );

		}

		//
		// event handlers - FSM: listen for events and reset state
		//

		function onPointerDown( event ) {

			if ( scope.enabled === false ) return;

			if ( pointers.length === 0 ) {

				scope.domElement.setPointerCapture( event.pointerId );

				scope.domElement.addEventListener( 'pointermove', onPointerMove );
				scope.domElement.addEventListener( 'pointerup', onPointerUp );

			}

			//

			if ( isTrackingPointer( event ) ) return;

			//

			addPointer( event );

			if ( event.pointerType === 'touch' ) {

				onTouchStart( event );

			} else {

				onMouseDown( event );

			}

		}

		function onPointerMove( event ) {

			if ( scope.enabled === false ) return;

			if ( event.pointerType === 'touch' ) {

				onTouchMove( event );

			} else {

				onMouseMove( event );

			}

		}

		function onPointerUp( event ) {

			removePointer( event );

			switch ( pointers.length ) {

				case 0:

					scope.domElement.releasePointerCapture( event.pointerId );

					scope.domElement.removeEventListener( 'pointermove', onPointerMove );
					scope.domElement.removeEventListener( 'pointerup', onPointerUp );

					scope.dispatchEvent( _endEvent );

					state = STATE.NONE;

					break;

				case 1:

					const pointerId = pointers[ 0 ];
					const position = pointerPositions[ pointerId ];

					// minimal placeholder event - allows state correction on pointer-up
					onTouchStart( { pointerId: pointerId, pageX: position.x, pageY: position.y } );

					break;

			}

		}

		function onMouseDown( event ) {

			let mouseAction;

			switch ( event.button ) {

				case 0:

					mouseAction = scope.mouseButtons.LEFT;
					break;

				case 1:

					mouseAction = scope.mouseButtons.MIDDLE;
					break;

				case 2:

					mouseAction = scope.mouseButtons.RIGHT;
					break;

				default:

					mouseAction = - 1;

			}

			switch ( mouseAction ) {

				case MOUSE.DOLLY:

					if ( scope.enableZoom === false ) return;

					handleMouseDownDolly( event );

					state = STATE.DOLLY;

					break;

				case MOUSE.ROTATE:

					if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

						if ( scope.enablePan === false ) return;

						handleMouseDownPan( event );

						state = STATE.PAN;

					} else {

						if ( scope.enableRotate === false ) return;

						handleMouseDownRotate( event );

						state = STATE.ROTATE;

					}

					break;

				case MOUSE.PAN:

					if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

						if ( scope.enableRotate === false ) return;

						handleMouseDownRotate( event );

						state = STATE.ROTATE;

					} else {

						if ( scope.enablePan === false ) return;

						handleMouseDownPan( event );

						state = STATE.PAN;

					}

					break;

				default:

					state = STATE.NONE;

			}

			if ( state !== STATE.NONE ) {

				scope.dispatchEvent( _startEvent );

			}

		}

		function onMouseMove( event ) {

			switch ( state ) {

				case STATE.ROTATE:

					if ( scope.enableRotate === false ) return;

					handleMouseMoveRotate( event );

					break;

				case STATE.DOLLY:

					if ( scope.enableZoom === false ) return;

					handleMouseMoveDolly( event );

					break;

				case STATE.PAN:

					if ( scope.enablePan === false ) return;

					handleMouseMovePan( event );

					break;

			}

		}

		function onMouseWheel( event ) {

			if ( scope.enabled === false || scope.enableZoom === false || state !== STATE.NONE ) return;

			event.preventDefault();

			scope.dispatchEvent( _startEvent );

			handleMouseWheel( customWheelEvent( event ) );

			scope.dispatchEvent( _endEvent );

		}

		function customWheelEvent( event ) {

			const mode = event.deltaMode;

			// minimal wheel event altered to meet delta-zoom demand
			const newEvent = {
				clientX: event.clientX,
				clientY: event.clientY,
				deltaY: event.deltaY,
			};

			switch ( mode ) {

				case 1: // LINE_MODE
					newEvent.deltaY *= 16;
					break;

				case 2: // PAGE_MODE
					newEvent.deltaY *= 100;
					break;

			}

			// detect if event was triggered by pinching
			if ( event.ctrlKey && ! controlActive ) {

				newEvent.deltaY *= 10;

			}

			return newEvent;

		}

		function interceptControlDown( event ) {

			if ( event.key === 'Control' ) {

				controlActive = true;


				const document = scope.domElement.getRootNode(); // offscreen canvas compatibility

				document.addEventListener( 'keyup', interceptControlUp, { passive: true, capture: true } );

			}

		}

		function interceptControlUp( event ) {

			if ( event.key === 'Control' ) {

				controlActive = false;


				const document = scope.domElement.getRootNode(); // offscreen canvas compatibility

				document.removeEventListener( 'keyup', interceptControlUp, { passive: true, capture: true } );

			}

		}

		function onKeyDown( event ) {

			if ( scope.enabled === false || scope.enablePan === false ) return;

			handleKeyDown( event );

		}

		function onTouchStart( event ) {

			trackPointer( event );

			switch ( pointers.length ) {

				case 1:

					switch ( scope.touches.ONE ) {

						case TOUCH.ROTATE:

							if ( scope.enableRotate === false ) return;

							handleTouchStartRotate( event );

							state = STATE.TOUCH_ROTATE;

							break;

						case TOUCH.PAN:

							if ( scope.enablePan === false ) return;

							handleTouchStartPan( event );

							state = STATE.TOUCH_PAN;

							break;

						default:

							state = STATE.NONE;

					}

					break;

				case 2:

					switch ( scope.touches.TWO ) {

						case TOUCH.DOLLY_PAN:

							if ( scope.enableZoom === false && scope.enablePan === false ) return;

							handleTouchStartDollyPan( event );

							state = STATE.TOUCH_DOLLY_PAN;

							break;

						case TOUCH.DOLLY_ROTATE:

							if ( scope.enableZoom === false && scope.enableRotate === false ) return;

							handleTouchStartDollyRotate( event );

							state = STATE.TOUCH_DOLLY_ROTATE;

							break;

						default:

							state = STATE.NONE;

					}

					break;

				default:

					state = STATE.NONE;

			}

			if ( state !== STATE.NONE ) {

				scope.dispatchEvent( _startEvent );

			}

		}

		function onTouchMove( event ) {

			trackPointer( event );

			switch ( state ) {

				case STATE.TOUCH_ROTATE:

					if ( scope.enableRotate === false ) return;

					handleTouchMoveRotate( event );

					scope.update();

					break;

				case STATE.TOUCH_PAN:

					if ( scope.enablePan === false ) return;

					handleTouchMovePan( event );

					scope.update();

					break;

				case STATE.TOUCH_DOLLY_PAN:

					if ( scope.enableZoom === false && scope.enablePan === false ) return;

					handleTouchMoveDollyPan( event );

					scope.update();

					break;

				case STATE.TOUCH_DOLLY_ROTATE:

					if ( scope.enableZoom === false && scope.enableRotate === false ) return;

					handleTouchMoveDollyRotate( event );

					scope.update();

					break;

				default:

					state = STATE.NONE;

			}

		}

		function onContextMenu( event ) {

			if ( scope.enabled === false ) return;

			event.preventDefault();

		}

		function addPointer( event ) {

			pointers.push( event.pointerId );

		}

		function removePointer( event ) {

			delete pointerPositions[ event.pointerId ];

			for ( let i = 0; i < pointers.length; i ++ ) {

				if ( pointers[ i ] == event.pointerId ) {

					pointers.splice( i, 1 );
					return;

				}

			}

		}

		function isTrackingPointer( event ) {

			for ( let i = 0; i < pointers.length; i ++ ) {

				if ( pointers[ i ] == event.pointerId ) return true;

			}

			return false;

		}

		function trackPointer( event ) {

			let position = pointerPositions[ event.pointerId ];

			if ( position === undefined ) {

				position = new Vector2();
				pointerPositions[ event.pointerId ] = position;

			}

			position.set( event.pageX, event.pageY );

		}

		function getSecondPointerPosition( event ) {

			const pointerId = ( event.pointerId === pointers[ 0 ] ) ? pointers[ 1 ] : pointers[ 0 ];

			return pointerPositions[ pointerId ];

		}

		//

		scope.domElement.addEventListener( 'contextmenu', onContextMenu );

		scope.domElement.addEventListener( 'pointerdown', onPointerDown );
		scope.domElement.addEventListener( 'pointercancel', onPointerUp );
		scope.domElement.addEventListener( 'wheel', onMouseWheel, { passive: false } );

		const document = scope.domElement.getRootNode(); // offscreen canvas compatibility

		document.addEventListener( 'keydown', interceptControlDown, { passive: true, capture: true } );

		// force an update at start

		this.update();

	}

}

/**
 * Full-screen textured quad shader
 */

const CopyShader = {

	name: 'CopyShader',

	uniforms: {

		'tDiffuse': { value: null },
		'opacity': { value: 1.0 }

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`

		uniform float opacity;

		uniform sampler2D tDiffuse;

		varying vec2 vUv;

		void main() {

			vec4 texel = texture2D( tDiffuse, vUv );
			gl_FragColor = opacity * texel;


		}`

};

class Pass {

	constructor() {

		this.isPass = true;

		// if set to true, the pass is processed by the composer
		this.enabled = true;

		// if set to true, the pass indicates to swap read and write buffer after rendering
		this.needsSwap = true;

		// if set to true, the pass clears its buffer before rendering
		this.clear = false;

		// if set to true, the result of the pass is rendered to screen. This is set automatically by EffectComposer.
		this.renderToScreen = false;

	}

	setSize( /* width, height */ ) {}

	render( /* renderer, writeBuffer, readBuffer, deltaTime, maskActive */ ) {

		console.error( 'THREE.Pass: .render() must be implemented in derived pass.' );

	}

	dispose() {}

}

// Helper for passes that need to fill the viewport with a single quad.

const _camera = new OrthographicCamera( - 1, 1, 1, - 1, 0, 1 );

// https://github.com/mrdoob/three.js/pull/21358

class FullscreenTriangleGeometry extends BufferGeometry {

	constructor() {

		super();

		this.setAttribute( 'position', new Float32BufferAttribute( [ - 1, 3, 0, - 1, - 1, 0, 3, - 1, 0 ], 3 ) );
		this.setAttribute( 'uv', new Float32BufferAttribute( [ 0, 2, 0, 0, 2, 0 ], 2 ) );

	}

}

const _geometry = new FullscreenTriangleGeometry();

class FullScreenQuad {

	constructor( material ) {

		this._mesh = new Mesh( _geometry, material );

	}

	dispose() {

		this._mesh.geometry.dispose();

	}

	render( renderer ) {

		renderer.render( this._mesh, _camera );

	}

	get material() {

		return this._mesh.material;

	}

	set material( value ) {

		this._mesh.material = value;

	}

}

class ShaderPass extends Pass {

	constructor( shader, textureID ) {

		super();

		this.textureID = ( textureID !== undefined ) ? textureID : 'tDiffuse';

		if ( shader instanceof ShaderMaterial ) {

			this.uniforms = shader.uniforms;

			this.material = shader;

		} else if ( shader ) {

			this.uniforms = UniformsUtils.clone( shader.uniforms );

			this.material = new ShaderMaterial( {

				name: ( shader.name !== undefined ) ? shader.name : 'unspecified',
				defines: Object.assign( {}, shader.defines ),
				uniforms: this.uniforms,
				vertexShader: shader.vertexShader,
				fragmentShader: shader.fragmentShader

			} );

		}

		this.fsQuad = new FullScreenQuad( this.material );

	}

	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {

		if ( this.uniforms[ this.textureID ] ) {

			this.uniforms[ this.textureID ].value = readBuffer.texture;

		}

		this.fsQuad.material = this.material;

		if ( this.renderToScreen ) {

			renderer.setRenderTarget( null );
			this.fsQuad.render( renderer );

		} else {

			renderer.setRenderTarget( writeBuffer );
			// TODO: Avoid using autoClear properties, see https://github.com/mrdoob/three.js/pull/15571#issuecomment-465669600
			if ( this.clear ) renderer.clear( renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil );
			this.fsQuad.render( renderer );

		}

	}

	dispose() {

		this.material.dispose();

		this.fsQuad.dispose();

	}

}

class MaskPass extends Pass {

	constructor( scene, camera ) {

		super();

		this.scene = scene;
		this.camera = camera;

		this.clear = true;
		this.needsSwap = false;

		this.inverse = false;

	}

	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {

		const context = renderer.getContext();
		const state = renderer.state;

		// don't update color or depth

		state.buffers.color.setMask( false );
		state.buffers.depth.setMask( false );

		// lock buffers

		state.buffers.color.setLocked( true );
		state.buffers.depth.setLocked( true );

		// set up stencil

		let writeValue, clearValue;

		if ( this.inverse ) {

			writeValue = 0;
			clearValue = 1;

		} else {

			writeValue = 1;
			clearValue = 0;

		}

		state.buffers.stencil.setTest( true );
		state.buffers.stencil.setOp( context.REPLACE, context.REPLACE, context.REPLACE );
		state.buffers.stencil.setFunc( context.ALWAYS, writeValue, 0xffffffff );
		state.buffers.stencil.setClear( clearValue );
		state.buffers.stencil.setLocked( true );

		// draw into the stencil buffer

		renderer.setRenderTarget( readBuffer );
		if ( this.clear ) renderer.clear();
		renderer.render( this.scene, this.camera );

		renderer.setRenderTarget( writeBuffer );
		if ( this.clear ) renderer.clear();
		renderer.render( this.scene, this.camera );

		// unlock color and depth buffer and make them writable for subsequent rendering/clearing

		state.buffers.color.setLocked( false );
		state.buffers.depth.setLocked( false );

		state.buffers.color.setMask( true );
		state.buffers.depth.setMask( true );

		// only render where stencil is set to 1

		state.buffers.stencil.setLocked( false );
		state.buffers.stencil.setFunc( context.EQUAL, 1, 0xffffffff ); // draw if == 1
		state.buffers.stencil.setOp( context.KEEP, context.KEEP, context.KEEP );
		state.buffers.stencil.setLocked( true );

	}

}

class ClearMaskPass extends Pass {

	constructor() {

		super();

		this.needsSwap = false;

	}

	render( renderer /*, writeBuffer, readBuffer, deltaTime, maskActive */ ) {

		renderer.state.buffers.stencil.setLocked( false );
		renderer.state.buffers.stencil.setTest( false );

	}

}

class EffectComposer {

	constructor( renderer, renderTarget ) {

		this.renderer = renderer;

		this._pixelRatio = renderer.getPixelRatio();

		if ( renderTarget === undefined ) {

			const size = renderer.getSize( new Vector2() );
			this._width = size.width;
			this._height = size.height;

			renderTarget = new WebGLRenderTarget( this._width * this._pixelRatio, this._height * this._pixelRatio, { type: HalfFloatType } );
			renderTarget.texture.name = 'EffectComposer.rt1';

		} else {

			this._width = renderTarget.width;
			this._height = renderTarget.height;

		}

		this.renderTarget1 = renderTarget;
		this.renderTarget2 = renderTarget.clone();
		this.renderTarget2.texture.name = 'EffectComposer.rt2';

		this.writeBuffer = this.renderTarget1;
		this.readBuffer = this.renderTarget2;

		this.renderToScreen = true;

		this.passes = [];

		this.copyPass = new ShaderPass( CopyShader );
		this.copyPass.material.blending = NoBlending;

		this.clock = new Clock();

	}

	swapBuffers() {

		const tmp = this.readBuffer;
		this.readBuffer = this.writeBuffer;
		this.writeBuffer = tmp;

	}

	addPass( pass ) {

		this.passes.push( pass );
		pass.setSize( this._width * this._pixelRatio, this._height * this._pixelRatio );

	}

	insertPass( pass, index ) {

		this.passes.splice( index, 0, pass );
		pass.setSize( this._width * this._pixelRatio, this._height * this._pixelRatio );

	}

	removePass( pass ) {

		const index = this.passes.indexOf( pass );

		if ( index !== - 1 ) {

			this.passes.splice( index, 1 );

		}

	}

	isLastEnabledPass( passIndex ) {

		for ( let i = passIndex + 1; i < this.passes.length; i ++ ) {

			if ( this.passes[ i ].enabled ) {

				return false;

			}

		}

		return true;

	}

	render( deltaTime ) {

		// deltaTime value is in seconds

		if ( deltaTime === undefined ) {

			deltaTime = this.clock.getDelta();

		}

		const currentRenderTarget = this.renderer.getRenderTarget();

		let maskActive = false;

		for ( let i = 0, il = this.passes.length; i < il; i ++ ) {

			const pass = this.passes[ i ];

			if ( pass.enabled === false ) continue;

			pass.renderToScreen = ( this.renderToScreen && this.isLastEnabledPass( i ) );
			pass.render( this.renderer, this.writeBuffer, this.readBuffer, deltaTime, maskActive );

			if ( pass.needsSwap ) {

				if ( maskActive ) {

					const context = this.renderer.getContext();
					const stencil = this.renderer.state.buffers.stencil;

					//context.stencilFunc( context.NOTEQUAL, 1, 0xffffffff );
					stencil.setFunc( context.NOTEQUAL, 1, 0xffffffff );

					this.copyPass.render( this.renderer, this.writeBuffer, this.readBuffer, deltaTime );

					//context.stencilFunc( context.EQUAL, 1, 0xffffffff );
					stencil.setFunc( context.EQUAL, 1, 0xffffffff );

				}

				this.swapBuffers();

			}

			if ( MaskPass !== undefined ) {

				if ( pass instanceof MaskPass ) {

					maskActive = true;

				} else if ( pass instanceof ClearMaskPass ) {

					maskActive = false;

				}

			}

		}

		this.renderer.setRenderTarget( currentRenderTarget );

	}

	reset( renderTarget ) {

		if ( renderTarget === undefined ) {

			const size = this.renderer.getSize( new Vector2() );
			this._pixelRatio = this.renderer.getPixelRatio();
			this._width = size.width;
			this._height = size.height;

			renderTarget = this.renderTarget1.clone();
			renderTarget.setSize( this._width * this._pixelRatio, this._height * this._pixelRatio );

		}

		this.renderTarget1.dispose();
		this.renderTarget2.dispose();
		this.renderTarget1 = renderTarget;
		this.renderTarget2 = renderTarget.clone();

		this.writeBuffer = this.renderTarget1;
		this.readBuffer = this.renderTarget2;

	}

	setSize( width, height ) {

		this._width = width;
		this._height = height;

		const effectiveWidth = this._width * this._pixelRatio;
		const effectiveHeight = this._height * this._pixelRatio;

		this.renderTarget1.setSize( effectiveWidth, effectiveHeight );
		this.renderTarget2.setSize( effectiveWidth, effectiveHeight );

		for ( let i = 0; i < this.passes.length; i ++ ) {

			this.passes[ i ].setSize( effectiveWidth, effectiveHeight );

		}

	}

	setPixelRatio( pixelRatio ) {

		this._pixelRatio = pixelRatio;

		this.setSize( this._width, this._height );

	}

	dispose() {

		this.renderTarget1.dispose();
		this.renderTarget2.dispose();

		this.copyPass.dispose();

	}

}

class RenderPass extends Pass {

	constructor( scene, camera, overrideMaterial = null, clearColor = null, clearAlpha = null ) {

		super();

		this.scene = scene;
		this.camera = camera;

		this.overrideMaterial = overrideMaterial;

		this.clearColor = clearColor;
		this.clearAlpha = clearAlpha;

		this.clear = true;
		this.clearDepth = false;
		this.needsSwap = false;
		this._oldClearColor = new Color();

	}

	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {

		const oldAutoClear = renderer.autoClear;
		renderer.autoClear = false;

		let oldClearAlpha, oldOverrideMaterial;

		if ( this.overrideMaterial !== null ) {

			oldOverrideMaterial = this.scene.overrideMaterial;

			this.scene.overrideMaterial = this.overrideMaterial;

		}

		if ( this.clearColor !== null ) {

			renderer.getClearColor( this._oldClearColor );
			renderer.setClearColor( this.clearColor );

		}

		if ( this.clearAlpha !== null ) {

			oldClearAlpha = renderer.getClearAlpha();
			renderer.setClearAlpha( this.clearAlpha );

		}

		if ( this.clearDepth == true ) {

			renderer.clearDepth();

		}

		renderer.setRenderTarget( this.renderToScreen ? null : readBuffer );

		if ( this.clear === true ) {

			// TODO: Avoid using autoClear properties, see https://github.com/mrdoob/three.js/pull/15571#issuecomment-465669600
			renderer.clear( renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil );

		}

		renderer.render( this.scene, this.camera );

		// restore

		if ( this.clearColor !== null ) {

			renderer.setClearColor( this._oldClearColor );

		}

		if ( this.clearAlpha !== null ) {

			renderer.setClearAlpha( oldClearAlpha );

		}

		if ( this.overrideMaterial !== null ) {

			this.scene.overrideMaterial = oldOverrideMaterial;

		}

		renderer.autoClear = oldAutoClear;

	}

}

// Ported from Stefan Gustavson's java implementation
// http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
// Read Stefan's excellent paper for details on how this code works.
//
// Sean McCullough banksean@gmail.com
//
// Added 4D noise

/**
 * You can pass in a random number generator object if you like.
 * It is assumed to have a random() method.
 */
class SimplexNoise {

	constructor( r = Math ) {

		this.grad3 = [[ 1, 1, 0 ], [ - 1, 1, 0 ], [ 1, - 1, 0 ], [ - 1, - 1, 0 ],
			[ 1, 0, 1 ], [ - 1, 0, 1 ], [ 1, 0, - 1 ], [ - 1, 0, - 1 ],
			[ 0, 1, 1 ], [ 0, - 1, 1 ], [ 0, 1, - 1 ], [ 0, - 1, - 1 ]];

		this.grad4 = [[ 0, 1, 1, 1 ], [ 0, 1, 1, - 1 ], [ 0, 1, - 1, 1 ], [ 0, 1, - 1, - 1 ],
			[ 0, - 1, 1, 1 ], [ 0, - 1, 1, - 1 ], [ 0, - 1, - 1, 1 ], [ 0, - 1, - 1, - 1 ],
			[ 1, 0, 1, 1 ], [ 1, 0, 1, - 1 ], [ 1, 0, - 1, 1 ], [ 1, 0, - 1, - 1 ],
			[ - 1, 0, 1, 1 ], [ - 1, 0, 1, - 1 ], [ - 1, 0, - 1, 1 ], [ - 1, 0, - 1, - 1 ],
			[ 1, 1, 0, 1 ], [ 1, 1, 0, - 1 ], [ 1, - 1, 0, 1 ], [ 1, - 1, 0, - 1 ],
			[ - 1, 1, 0, 1 ], [ - 1, 1, 0, - 1 ], [ - 1, - 1, 0, 1 ], [ - 1, - 1, 0, - 1 ],
			[ 1, 1, 1, 0 ], [ 1, 1, - 1, 0 ], [ 1, - 1, 1, 0 ], [ 1, - 1, - 1, 0 ],
			[ - 1, 1, 1, 0 ], [ - 1, 1, - 1, 0 ], [ - 1, - 1, 1, 0 ], [ - 1, - 1, - 1, 0 ]];

		this.p = [];

		for ( let i = 0; i < 256; i ++ ) {

			this.p[ i ] = Math.floor( r.random() * 256 );

		}

		// To remove the need for index wrapping, double the permutation table length
		this.perm = [];

		for ( let i = 0; i < 512; i ++ ) {

			this.perm[ i ] = this.p[ i & 255 ];

		}

		// A lookup table to traverse the simplex around a given point in 4D.
		// Details can be found where this table is used, in the 4D noise method.
		this.simplex = [
			[ 0, 1, 2, 3 ], [ 0, 1, 3, 2 ], [ 0, 0, 0, 0 ], [ 0, 2, 3, 1 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 1, 2, 3, 0 ],
			[ 0, 2, 1, 3 ], [ 0, 0, 0, 0 ], [ 0, 3, 1, 2 ], [ 0, 3, 2, 1 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 1, 3, 2, 0 ],
			[ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ],
			[ 1, 2, 0, 3 ], [ 0, 0, 0, 0 ], [ 1, 3, 0, 2 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 2, 3, 0, 1 ], [ 2, 3, 1, 0 ],
			[ 1, 0, 2, 3 ], [ 1, 0, 3, 2 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 2, 0, 3, 1 ], [ 0, 0, 0, 0 ], [ 2, 1, 3, 0 ],
			[ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ],
			[ 2, 0, 1, 3 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 3, 0, 1, 2 ], [ 3, 0, 2, 1 ], [ 0, 0, 0, 0 ], [ 3, 1, 2, 0 ],
			[ 2, 1, 0, 3 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 3, 1, 0, 2 ], [ 0, 0, 0, 0 ], [ 3, 2, 0, 1 ], [ 3, 2, 1, 0 ]];

	}

	dot( g, x, y ) {

		return g[ 0 ] * x + g[ 1 ] * y;

	}

	dot3( g, x, y, z ) {

		return g[ 0 ] * x + g[ 1 ] * y + g[ 2 ] * z;

	}

	dot4( g, x, y, z, w ) {

		return g[ 0 ] * x + g[ 1 ] * y + g[ 2 ] * z + g[ 3 ] * w;

	}

	noise( xin, yin ) {

		let n0; // Noise contributions from the three corners
		let n1;
		let n2;
		// Skew the input space to determine which simplex cell we're in
		const F2 = 0.5 * ( Math.sqrt( 3.0 ) - 1.0 );
		const s = ( xin + yin ) * F2; // Hairy factor for 2D
		const i = Math.floor( xin + s );
		const j = Math.floor( yin + s );
		const G2 = ( 3.0 - Math.sqrt( 3.0 ) ) / 6.0;
		const t = ( i + j ) * G2;
		const X0 = i - t; // Unskew the cell origin back to (x,y) space
		const Y0 = j - t;
		const x0 = xin - X0; // The x,y distances from the cell origin
		const y0 = yin - Y0;

		// For the 2D case, the simplex shape is an equilateral triangle.
		// Determine which simplex we are in.
		let i1; // Offsets for second (middle) corner of simplex in (i,j) coords

		let j1;
		if ( x0 > y0 ) {

			i1 = 1; j1 = 0;

			// lower triangle, XY order: (0,0)->(1,0)->(1,1)

		}	else {

			i1 = 0; j1 = 1;

		} // upper triangle, YX order: (0,0)->(0,1)->(1,1)

		// A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
		// a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
		// c = (3-sqrt(3))/6
		const x1 = x0 - i1 + G2; // Offsets for middle corner in (x,y) unskewed coords
		const y1 = y0 - j1 + G2;
		const x2 = x0 - 1.0 + 2.0 * G2; // Offsets for last corner in (x,y) unskewed coords
		const y2 = y0 - 1.0 + 2.0 * G2;
		// Work out the hashed gradient indices of the three simplex corners
		const ii = i & 255;
		const jj = j & 255;
		const gi0 = this.perm[ ii + this.perm[ jj ] ] % 12;
		const gi1 = this.perm[ ii + i1 + this.perm[ jj + j1 ] ] % 12;
		const gi2 = this.perm[ ii + 1 + this.perm[ jj + 1 ] ] % 12;
		// Calculate the contribution from the three corners
		let t0 = 0.5 - x0 * x0 - y0 * y0;
		if ( t0 < 0 ) n0 = 0.0;
		else {

			t0 *= t0;
			n0 = t0 * t0 * this.dot( this.grad3[ gi0 ], x0, y0 ); // (x,y) of grad3 used for 2D gradient

		}

		let t1 = 0.5 - x1 * x1 - y1 * y1;
		if ( t1 < 0 ) n1 = 0.0;
		else {

			t1 *= t1;
			n1 = t1 * t1 * this.dot( this.grad3[ gi1 ], x1, y1 );

		}

		let t2 = 0.5 - x2 * x2 - y2 * y2;
		if ( t2 < 0 ) n2 = 0.0;
		else {

			t2 *= t2;
			n2 = t2 * t2 * this.dot( this.grad3[ gi2 ], x2, y2 );

		}

		// Add contributions from each corner to get the final noise value.
		// The result is scaled to return values in the interval [-1,1].
		return 70.0 * ( n0 + n1 + n2 );

	}

	// 3D simplex noise
	noise3d( xin, yin, zin ) {

		let n0; // Noise contributions from the four corners
		let n1;
		let n2;
		let n3;
		// Skew the input space to determine which simplex cell we're in
		const F3 = 1.0 / 3.0;
		const s = ( xin + yin + zin ) * F3; // Very nice and simple skew factor for 3D
		const i = Math.floor( xin + s );
		const j = Math.floor( yin + s );
		const k = Math.floor( zin + s );
		const G3 = 1.0 / 6.0; // Very nice and simple unskew factor, too
		const t = ( i + j + k ) * G3;
		const X0 = i - t; // Unskew the cell origin back to (x,y,z) space
		const Y0 = j - t;
		const Z0 = k - t;
		const x0 = xin - X0; // The x,y,z distances from the cell origin
		const y0 = yin - Y0;
		const z0 = zin - Z0;

		// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
		// Determine which simplex we are in.
		let i1; // Offsets for second corner of simplex in (i,j,k) coords

		let j1;
		let k1;
		let i2; // Offsets for third corner of simplex in (i,j,k) coords
		let j2;
		let k2;
		if ( x0 >= y0 ) {

			if ( y0 >= z0 ) {

				i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;

				// X Y Z order

			} else if ( x0 >= z0 ) {

				i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;

				// X Z Y order

			} else {

				i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;

			} // Z X Y order

		} else { // x0<y0

			if ( y0 < z0 ) {

				i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;

				// Z Y X order

			} else if ( x0 < z0 ) {

				i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;

				// Y Z X order

			} else {

				i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;

			} // Y X Z order

		}

		// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
		// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
		// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
		// c = 1/6.
		const x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
		const y1 = y0 - j1 + G3;
		const z1 = z0 - k1 + G3;
		const x2 = x0 - i2 + 2.0 * G3; // Offsets for third corner in (x,y,z) coords
		const y2 = y0 - j2 + 2.0 * G3;
		const z2 = z0 - k2 + 2.0 * G3;
		const x3 = x0 - 1.0 + 3.0 * G3; // Offsets for last corner in (x,y,z) coords
		const y3 = y0 - 1.0 + 3.0 * G3;
		const z3 = z0 - 1.0 + 3.0 * G3;
		// Work out the hashed gradient indices of the four simplex corners
		const ii = i & 255;
		const jj = j & 255;
		const kk = k & 255;
		const gi0 = this.perm[ ii + this.perm[ jj + this.perm[ kk ] ] ] % 12;
		const gi1 = this.perm[ ii + i1 + this.perm[ jj + j1 + this.perm[ kk + k1 ] ] ] % 12;
		const gi2 = this.perm[ ii + i2 + this.perm[ jj + j2 + this.perm[ kk + k2 ] ] ] % 12;
		const gi3 = this.perm[ ii + 1 + this.perm[ jj + 1 + this.perm[ kk + 1 ] ] ] % 12;
		// Calculate the contribution from the four corners
		let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
		if ( t0 < 0 ) n0 = 0.0;
		else {

			t0 *= t0;
			n0 = t0 * t0 * this.dot3( this.grad3[ gi0 ], x0, y0, z0 );

		}

		let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
		if ( t1 < 0 ) n1 = 0.0;
		else {

			t1 *= t1;
			n1 = t1 * t1 * this.dot3( this.grad3[ gi1 ], x1, y1, z1 );

		}

		let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
		if ( t2 < 0 ) n2 = 0.0;
		else {

			t2 *= t2;
			n2 = t2 * t2 * this.dot3( this.grad3[ gi2 ], x2, y2, z2 );

		}

		let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
		if ( t3 < 0 ) n3 = 0.0;
		else {

			t3 *= t3;
			n3 = t3 * t3 * this.dot3( this.grad3[ gi3 ], x3, y3, z3 );

		}

		// Add contributions from each corner to get the final noise value.
		// The result is scaled to stay just inside [-1,1]
		return 32.0 * ( n0 + n1 + n2 + n3 );

	}

	// 4D simplex noise
	noise4d( x, y, z, w ) {

		// For faster and easier lookups
		const grad4 = this.grad4;
		const simplex = this.simplex;
		const perm = this.perm;

		// The skewing and unskewing factors are hairy again for the 4D case
		const F4 = ( Math.sqrt( 5.0 ) - 1.0 ) / 4.0;
		const G4 = ( 5.0 - Math.sqrt( 5.0 ) ) / 20.0;
		let n0; // Noise contributions from the five corners
		let n1;
		let n2;
		let n3;
		let n4;
		// Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
		const s = ( x + y + z + w ) * F4; // Factor for 4D skewing
		const i = Math.floor( x + s );
		const j = Math.floor( y + s );
		const k = Math.floor( z + s );
		const l = Math.floor( w + s );
		const t = ( i + j + k + l ) * G4; // Factor for 4D unskewing
		const X0 = i - t; // Unskew the cell origin back to (x,y,z,w) space
		const Y0 = j - t;
		const Z0 = k - t;
		const W0 = l - t;
		const x0 = x - X0; // The x,y,z,w distances from the cell origin
		const y0 = y - Y0;
		const z0 = z - Z0;
		const w0 = w - W0;

		// For the 4D case, the simplex is a 4D shape I won't even try to describe.
		// To find out which of the 24 possible simplices we're in, we need to
		// determine the magnitude ordering of x0, y0, z0 and w0.
		// The method below is a good way of finding the ordering of x,y,z,w and
		// then find the correct traversal order for the simplex we’re in.
		// First, six pair-wise comparisons are performed between each possible pair
		// of the four coordinates, and the results are used to add up binary bits
		// for an integer index.
		const c1 = ( x0 > y0 ) ? 32 : 0;
		const c2 = ( x0 > z0 ) ? 16 : 0;
		const c3 = ( y0 > z0 ) ? 8 : 0;
		const c4 = ( x0 > w0 ) ? 4 : 0;
		const c5 = ( y0 > w0 ) ? 2 : 0;
		const c6 = ( z0 > w0 ) ? 1 : 0;
		const c = c1 + c2 + c3 + c4 + c5 + c6;

		// simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
		// Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
		// impossible. Only the 24 indices which have non-zero entries make any sense.
		// We use a thresholding to set the coordinates in turn from the largest magnitude.
		// The number 3 in the "simplex" array is at the position of the largest coordinate.
		const i1 = simplex[ c ][ 0 ] >= 3 ? 1 : 0;
		const j1 = simplex[ c ][ 1 ] >= 3 ? 1 : 0;
		const k1 = simplex[ c ][ 2 ] >= 3 ? 1 : 0;
		const l1 = simplex[ c ][ 3 ] >= 3 ? 1 : 0;
		// The number 2 in the "simplex" array is at the second largest coordinate.
		const i2 = simplex[ c ][ 0 ] >= 2 ? 1 : 0;
		const j2 = simplex[ c ][ 1 ] >= 2 ? 1 : 0;
		const k2 = simplex[ c ][ 2 ] >= 2 ? 1 : 0;
		const l2 = simplex[ c ][ 3 ] >= 2 ? 1 : 0;
		// The number 1 in the "simplex" array is at the second smallest coordinate.
		const i3 = simplex[ c ][ 0 ] >= 1 ? 1 : 0;
		const j3 = simplex[ c ][ 1 ] >= 1 ? 1 : 0;
		const k3 = simplex[ c ][ 2 ] >= 1 ? 1 : 0;
		const l3 = simplex[ c ][ 3 ] >= 1 ? 1 : 0;
		// The fifth corner has all coordinate offsets = 1, so no need to look that up.
		const x1 = x0 - i1 + G4; // Offsets for second corner in (x,y,z,w) coords
		const y1 = y0 - j1 + G4;
		const z1 = z0 - k1 + G4;
		const w1 = w0 - l1 + G4;
		const x2 = x0 - i2 + 2.0 * G4; // Offsets for third corner in (x,y,z,w) coords
		const y2 = y0 - j2 + 2.0 * G4;
		const z2 = z0 - k2 + 2.0 * G4;
		const w2 = w0 - l2 + 2.0 * G4;
		const x3 = x0 - i3 + 3.0 * G4; // Offsets for fourth corner in (x,y,z,w) coords
		const y3 = y0 - j3 + 3.0 * G4;
		const z3 = z0 - k3 + 3.0 * G4;
		const w3 = w0 - l3 + 3.0 * G4;
		const x4 = x0 - 1.0 + 4.0 * G4; // Offsets for last corner in (x,y,z,w) coords
		const y4 = y0 - 1.0 + 4.0 * G4;
		const z4 = z0 - 1.0 + 4.0 * G4;
		const w4 = w0 - 1.0 + 4.0 * G4;
		// Work out the hashed gradient indices of the five simplex corners
		const ii = i & 255;
		const jj = j & 255;
		const kk = k & 255;
		const ll = l & 255;
		const gi0 = perm[ ii + perm[ jj + perm[ kk + perm[ ll ] ] ] ] % 32;
		const gi1 = perm[ ii + i1 + perm[ jj + j1 + perm[ kk + k1 + perm[ ll + l1 ] ] ] ] % 32;
		const gi2 = perm[ ii + i2 + perm[ jj + j2 + perm[ kk + k2 + perm[ ll + l2 ] ] ] ] % 32;
		const gi3 = perm[ ii + i3 + perm[ jj + j3 + perm[ kk + k3 + perm[ ll + l3 ] ] ] ] % 32;
		const gi4 = perm[ ii + 1 + perm[ jj + 1 + perm[ kk + 1 + perm[ ll + 1 ] ] ] ] % 32;
		// Calculate the contribution from the five corners
		let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0;
		if ( t0 < 0 ) n0 = 0.0;
		else {

			t0 *= t0;
			n0 = t0 * t0 * this.dot4( grad4[ gi0 ], x0, y0, z0, w0 );

		}

		let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1;
		if ( t1 < 0 ) n1 = 0.0;
		else {

			t1 *= t1;
			n1 = t1 * t1 * this.dot4( grad4[ gi1 ], x1, y1, z1, w1 );

		}

		let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2;
		if ( t2 < 0 ) n2 = 0.0;
		else {

			t2 *= t2;
			n2 = t2 * t2 * this.dot4( grad4[ gi2 ], x2, y2, z2, w2 );

		}

		let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3;
		if ( t3 < 0 ) n3 = 0.0;
		else {

			t3 *= t3;
			n3 = t3 * t3 * this.dot4( grad4[ gi3 ], x3, y3, z3, w3 );

		}

		let t4 = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4;
		if ( t4 < 0 ) n4 = 0.0;
		else {

			t4 *= t4;
			n4 = t4 * t4 * this.dot4( grad4[ gi4 ], x4, y4, z4, w4 );

		}

		// Sum up and scale the result to cover the range [-1,1]
		return 27.0 * ( n0 + n1 + n2 + n3 + n4 );

	}

}

/**
 * Luminosity
 * http://en.wikipedia.org/wiki/Luminosity
 */

const LuminosityHighPassShader = {

	name: 'LuminosityHighPassShader',

	shaderID: 'luminosityHighPass',

	uniforms: {

		'tDiffuse': { value: null },
		'luminosityThreshold': { value: 1.0 },
		'smoothWidth': { value: 1.0 },
		'defaultColor': { value: new Color( 0x000000 ) },
		'defaultOpacity': { value: 0.0 }

	},

	vertexShader: /* glsl */`

		varying vec2 vUv;

		void main() {

			vUv = uv;

			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );

		}`,

	fragmentShader: /* glsl */`

		uniform sampler2D tDiffuse;
		uniform vec3 defaultColor;
		uniform float defaultOpacity;
		uniform float luminosityThreshold;
		uniform float smoothWidth;

		varying vec2 vUv;

		void main() {

			vec4 texel = texture2D( tDiffuse, vUv );

			vec3 luma = vec3( 0.299, 0.587, 0.114 );

			float v = dot( texel.xyz, luma );

			vec4 outputColor = vec4( defaultColor.rgb, defaultOpacity );

			float alpha = smoothstep( luminosityThreshold, luminosityThreshold + smoothWidth, v );

			gl_FragColor = mix( outputColor, texel, alpha );

		}`

};

/**
 * UnrealBloomPass is inspired by the bloom pass of Unreal Engine. It creates a
 * mip map chain of bloom textures and blurs them with different radii. Because
 * of the weighted combination of mips, and because larger blurs are done on
 * higher mips, this effect provides good quality and performance.
 *
 * Reference:
 * - https://docs.unrealengine.com/latest/INT/Engine/Rendering/PostProcessEffects/Bloom/
 */
class UnrealBloomPass extends Pass {

	constructor( resolution, strength, radius, threshold ) {

		super();

		this.strength = ( strength !== undefined ) ? strength : 1;
		this.radius = radius;
		this.threshold = threshold;
		this.resolution = ( resolution !== undefined ) ? new Vector2( resolution.x, resolution.y ) : new Vector2( 256, 256 );

		// create color only once here, reuse it later inside the render function
		this.clearColor = new Color( 0, 0, 0 );

		// render targets
		this.renderTargetsHorizontal = [];
		this.renderTargetsVertical = [];
		this.nMips = 5;
		let resx = Math.round( this.resolution.x / 2 );
		let resy = Math.round( this.resolution.y / 2 );

		this.renderTargetBright = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );
		this.renderTargetBright.texture.name = 'UnrealBloomPass.bright';
		this.renderTargetBright.texture.generateMipmaps = false;

		for ( let i = 0; i < this.nMips; i ++ ) {

			const renderTargetHorizonal = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );

			renderTargetHorizonal.texture.name = 'UnrealBloomPass.h' + i;
			renderTargetHorizonal.texture.generateMipmaps = false;

			this.renderTargetsHorizontal.push( renderTargetHorizonal );

			const renderTargetVertical = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );

			renderTargetVertical.texture.name = 'UnrealBloomPass.v' + i;
			renderTargetVertical.texture.generateMipmaps = false;

			this.renderTargetsVertical.push( renderTargetVertical );

			resx = Math.round( resx / 2 );

			resy = Math.round( resy / 2 );

		}

		// luminosity high pass material

		const highPassShader = LuminosityHighPassShader;
		this.highPassUniforms = UniformsUtils.clone( highPassShader.uniforms );

		this.highPassUniforms[ 'luminosityThreshold' ].value = threshold;
		this.highPassUniforms[ 'smoothWidth' ].value = 0.01;

		this.materialHighPassFilter = new ShaderMaterial( {
			uniforms: this.highPassUniforms,
			vertexShader: highPassShader.vertexShader,
			fragmentShader: highPassShader.fragmentShader
		} );

		// gaussian blur materials

		this.separableBlurMaterials = [];
		const kernelSizeArray = [ 3, 5, 7, 9, 11 ];
		resx = Math.round( this.resolution.x / 2 );
		resy = Math.round( this.resolution.y / 2 );

		for ( let i = 0; i < this.nMips; i ++ ) {

			this.separableBlurMaterials.push( this.getSeperableBlurMaterial( kernelSizeArray[ i ] ) );

			this.separableBlurMaterials[ i ].uniforms[ 'invSize' ].value = new Vector2( 1 / resx, 1 / resy );

			resx = Math.round( resx / 2 );

			resy = Math.round( resy / 2 );

		}

		// composite material

		this.compositeMaterial = this.getCompositeMaterial( this.nMips );
		this.compositeMaterial.uniforms[ 'blurTexture1' ].value = this.renderTargetsVertical[ 0 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture2' ].value = this.renderTargetsVertical[ 1 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture3' ].value = this.renderTargetsVertical[ 2 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture4' ].value = this.renderTargetsVertical[ 3 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture5' ].value = this.renderTargetsVertical[ 4 ].texture;
		this.compositeMaterial.uniforms[ 'bloomStrength' ].value = strength;
		this.compositeMaterial.uniforms[ 'bloomRadius' ].value = 0.1;

		const bloomFactors = [ 1.0, 0.8, 0.6, 0.4, 0.2 ];
		this.compositeMaterial.uniforms[ 'bloomFactors' ].value = bloomFactors;
		this.bloomTintColors = [ new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ) ];
		this.compositeMaterial.uniforms[ 'bloomTintColors' ].value = this.bloomTintColors;

		// blend material

		const copyShader = CopyShader;

		this.copyUniforms = UniformsUtils.clone( copyShader.uniforms );

		this.blendMaterial = new ShaderMaterial( {
			uniforms: this.copyUniforms,
			vertexShader: copyShader.vertexShader,
			fragmentShader: copyShader.fragmentShader,
			blending: AdditiveBlending,
			depthTest: false,
			depthWrite: false,
			transparent: true
		} );

		this.enabled = true;
		this.needsSwap = false;

		this._oldClearColor = new Color();
		this.oldClearAlpha = 1;

		this.basic = new MeshBasicMaterial();

		this.fsQuad = new FullScreenQuad( null );

	}

	dispose() {

		for ( let i = 0; i < this.renderTargetsHorizontal.length; i ++ ) {

			this.renderTargetsHorizontal[ i ].dispose();

		}

		for ( let i = 0; i < this.renderTargetsVertical.length; i ++ ) {

			this.renderTargetsVertical[ i ].dispose();

		}

		this.renderTargetBright.dispose();

		//

		for ( let i = 0; i < this.separableBlurMaterials.length; i ++ ) {

			this.separableBlurMaterials[ i ].dispose();

		}

		this.compositeMaterial.dispose();
		this.blendMaterial.dispose();
		this.basic.dispose();

		//

		this.fsQuad.dispose();

	}

	setSize( width, height ) {

		let resx = Math.round( width / 2 );
		let resy = Math.round( height / 2 );

		this.renderTargetBright.setSize( resx, resy );

		for ( let i = 0; i < this.nMips; i ++ ) {

			this.renderTargetsHorizontal[ i ].setSize( resx, resy );
			this.renderTargetsVertical[ i ].setSize( resx, resy );

			this.separableBlurMaterials[ i ].uniforms[ 'invSize' ].value = new Vector2( 1 / resx, 1 / resy );

			resx = Math.round( resx / 2 );
			resy = Math.round( resy / 2 );

		}

	}

	render( renderer, writeBuffer, readBuffer, deltaTime, maskActive ) {

		renderer.getClearColor( this._oldClearColor );
		this.oldClearAlpha = renderer.getClearAlpha();
		const oldAutoClear = renderer.autoClear;
		renderer.autoClear = false;

		renderer.setClearColor( this.clearColor, 0 );

		if ( maskActive ) renderer.state.buffers.stencil.setTest( false );

		// Render input to screen

		if ( this.renderToScreen ) {

			this.fsQuad.material = this.basic;
			this.basic.map = readBuffer.texture;

			renderer.setRenderTarget( null );
			renderer.clear();
			this.fsQuad.render( renderer );

		}

		// 1. Extract Bright Areas

		this.highPassUniforms[ 'tDiffuse' ].value = readBuffer.texture;
		this.highPassUniforms[ 'luminosityThreshold' ].value = this.threshold;
		this.fsQuad.material = this.materialHighPassFilter;

		renderer.setRenderTarget( this.renderTargetBright );
		renderer.clear();
		this.fsQuad.render( renderer );

		// 2. Blur All the mips progressively

		let inputRenderTarget = this.renderTargetBright;

		for ( let i = 0; i < this.nMips; i ++ ) {

			this.fsQuad.material = this.separableBlurMaterials[ i ];

			this.separableBlurMaterials[ i ].uniforms[ 'colorTexture' ].value = inputRenderTarget.texture;
			this.separableBlurMaterials[ i ].uniforms[ 'direction' ].value = UnrealBloomPass.BlurDirectionX;
			renderer.setRenderTarget( this.renderTargetsHorizontal[ i ] );
			renderer.clear();
			this.fsQuad.render( renderer );

			this.separableBlurMaterials[ i ].uniforms[ 'colorTexture' ].value = this.renderTargetsHorizontal[ i ].texture;
			this.separableBlurMaterials[ i ].uniforms[ 'direction' ].value = UnrealBloomPass.BlurDirectionY;
			renderer.setRenderTarget( this.renderTargetsVertical[ i ] );
			renderer.clear();
			this.fsQuad.render( renderer );

			inputRenderTarget = this.renderTargetsVertical[ i ];

		}

		// Composite All the mips

		this.fsQuad.material = this.compositeMaterial;
		this.compositeMaterial.uniforms[ 'bloomStrength' ].value = this.strength;
		this.compositeMaterial.uniforms[ 'bloomRadius' ].value = this.radius;
		this.compositeMaterial.uniforms[ 'bloomTintColors' ].value = this.bloomTintColors;

		renderer.setRenderTarget( this.renderTargetsHorizontal[ 0 ] );
		renderer.clear();
		this.fsQuad.render( renderer );

		// Blend it additively over the input texture

		this.fsQuad.material = this.blendMaterial;
		this.copyUniforms[ 'tDiffuse' ].value = this.renderTargetsHorizontal[ 0 ].texture;

		if ( maskActive ) renderer.state.buffers.stencil.setTest( true );

		if ( this.renderToScreen ) {

			renderer.setRenderTarget( null );
			this.fsQuad.render( renderer );

		} else {

			renderer.setRenderTarget( readBuffer );
			this.fsQuad.render( renderer );

		}

		// Restore renderer settings

		renderer.setClearColor( this._oldClearColor, this.oldClearAlpha );
		renderer.autoClear = oldAutoClear;

	}

	getSeperableBlurMaterial( kernelRadius ) {

		const coefficients = [];

		for ( let i = 0; i < kernelRadius; i ++ ) {

			coefficients.push( 0.39894 * Math.exp( - 0.5 * i * i / ( kernelRadius * kernelRadius ) ) / kernelRadius );

		}

		return new ShaderMaterial( {

			defines: {
				'KERNEL_RADIUS': kernelRadius
			},

			uniforms: {
				'colorTexture': { value: null },
				'invSize': { value: new Vector2( 0.5, 0.5 ) }, // inverse texture size
				'direction': { value: new Vector2( 0.5, 0.5 ) },
				'gaussianCoefficients': { value: coefficients } // precomputed Gaussian coefficients
			},

			vertexShader:
				`varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,

			fragmentShader:
				`#include <common>
				varying vec2 vUv;
				uniform sampler2D colorTexture;
				uniform vec2 invSize;
				uniform vec2 direction;
				uniform float gaussianCoefficients[KERNEL_RADIUS];

				void main() {
					float weightSum = gaussianCoefficients[0];
					vec3 diffuseSum = texture2D( colorTexture, vUv ).rgb * weightSum;
					for( int i = 1; i < KERNEL_RADIUS; i ++ ) {
						float x = float(i);
						float w = gaussianCoefficients[i];
						vec2 uvOffset = direction * invSize * x;
						vec3 sample1 = texture2D( colorTexture, vUv + uvOffset ).rgb;
						vec3 sample2 = texture2D( colorTexture, vUv - uvOffset ).rgb;
						diffuseSum += (sample1 + sample2) * w;
						weightSum += 2.0 * w;
					}
					gl_FragColor = vec4(diffuseSum/weightSum, 1.0);
				}`
		} );

	}

	getCompositeMaterial( nMips ) {

		return new ShaderMaterial( {

			defines: {
				'NUM_MIPS': nMips
			},

			uniforms: {
				'blurTexture1': { value: null },
				'blurTexture2': { value: null },
				'blurTexture3': { value: null },
				'blurTexture4': { value: null },
				'blurTexture5': { value: null },
				'bloomStrength': { value: 1.0 },
				'bloomFactors': { value: null },
				'bloomTintColors': { value: null },
				'bloomRadius': { value: 0.0 }
			},

			vertexShader:
				`varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,

			fragmentShader:
				`varying vec2 vUv;
				uniform sampler2D blurTexture1;
				uniform sampler2D blurTexture2;
				uniform sampler2D blurTexture3;
				uniform sampler2D blurTexture4;
				uniform sampler2D blurTexture5;
				uniform float bloomStrength;
				uniform float bloomRadius;
				uniform float bloomFactors[NUM_MIPS];
				uniform vec3 bloomTintColors[NUM_MIPS];

				float lerpBloomFactor(const in float factor) {
					float mirrorFactor = 1.2 - factor;
					return mix(factor, mirrorFactor, bloomRadius);
				}

				void main() {
					gl_FragColor = bloomStrength * ( lerpBloomFactor(bloomFactors[0]) * vec4(bloomTintColors[0], 1.0) * texture2D(blurTexture1, vUv) +
						lerpBloomFactor(bloomFactors[1]) * vec4(bloomTintColors[1], 1.0) * texture2D(blurTexture2, vUv) +
						lerpBloomFactor(bloomFactors[2]) * vec4(bloomTintColors[2], 1.0) * texture2D(blurTexture3, vUv) +
						lerpBloomFactor(bloomFactors[3]) * vec4(bloomTintColors[3], 1.0) * texture2D(blurTexture4, vUv) +
						lerpBloomFactor(bloomFactors[4]) * vec4(bloomTintColors[4], 1.0) * texture2D(blurTexture5, vUv) );
				}`
		} );

	}

}

UnrealBloomPass.BlurDirectionX = new Vector2( 1.0, 0.0 );
UnrealBloomPass.BlurDirectionY = new Vector2( 0.0, 1.0 );

class RenderableObject {

	constructor() {

		this.id = 0;

		this.object = null;
		this.z = 0;
		this.renderOrder = 0;

	}

}

//

class RenderableFace {

	constructor() {

		this.id = 0;

		this.v1 = new RenderableVertex();
		this.v2 = new RenderableVertex();
		this.v3 = new RenderableVertex();

		this.normalModel = new Vector3();

		this.vertexNormalsModel = [ new Vector3(), new Vector3(), new Vector3() ];
		this.vertexNormalsLength = 0;

		this.color = new Color();
		this.material = null;
		this.uvs = [ new Vector2(), new Vector2(), new Vector2() ];

		this.z = 0;
		this.renderOrder = 0;

	}

}

//

class RenderableVertex {

	constructor() {

		this.position = new Vector3();
		this.positionWorld = new Vector3();
		this.positionScreen = new Vector4();

		this.visible = true;

	}

	copy( vertex ) {

		this.positionWorld.copy( vertex.positionWorld );
		this.positionScreen.copy( vertex.positionScreen );

	}

}

//

class RenderableLine {

	constructor() {

		this.id = 0;

		this.v1 = new RenderableVertex();
		this.v2 = new RenderableVertex();

		this.vertexColors = [ new Color(), new Color() ];
		this.material = null;

		this.z = 0;
		this.renderOrder = 0;

	}

}

//

class RenderableSprite {

	constructor() {

		this.id = 0;

		this.object = null;

		this.x = 0;
		this.y = 0;
		this.z = 0;

		this.rotation = 0;
		this.scale = new Vector2();

		this.material = null;
		this.renderOrder = 0;

	}

}

//

class Projector {

	constructor() {

		let _object, _objectCount, _objectPoolLength = 0,
			_vertex, _vertexCount, _vertexPoolLength = 0,
			_face, _faceCount, _facePoolLength = 0,
			_line, _lineCount, _linePoolLength = 0,
			_sprite, _spriteCount, _spritePoolLength = 0,
			_modelMatrix;

		const

			_renderData = { objects: [], lights: [], elements: [] },

			_vector3 = new Vector3(),
			_vector4 = new Vector4(),

			_clipBox = new Box3( new Vector3( - 1, - 1, - 1 ), new Vector3( 1, 1, 1 ) ),
			_boundingBox = new Box3(),
			_points3 = new Array( 3 ),

			_viewMatrix = new Matrix4(),
			_viewProjectionMatrix = new Matrix4(),

			_modelViewProjectionMatrix = new Matrix4(),

			_frustum = new Frustum(),

			_objectPool = [], _vertexPool = [], _facePool = [], _linePool = [], _spritePool = [];

		//

		function RenderList() {

			const normals = [];
			const colors = [];
			const uvs = [];

			let object = null;

			const normalMatrix = new Matrix3();

			function setObject( value ) {

				object = value;

				normalMatrix.getNormalMatrix( object.matrixWorld );

				normals.length = 0;
				colors.length = 0;
				uvs.length = 0;

			}

			function projectVertex( vertex ) {

				const position = vertex.position;
				const positionWorld = vertex.positionWorld;
				const positionScreen = vertex.positionScreen;

				positionWorld.copy( position ).applyMatrix4( _modelMatrix );
				positionScreen.copy( positionWorld ).applyMatrix4( _viewProjectionMatrix );

				const invW = 1 / positionScreen.w;

				positionScreen.x *= invW;
				positionScreen.y *= invW;
				positionScreen.z *= invW;

				vertex.visible = positionScreen.x >= - 1 && positionScreen.x <= 1 &&
						 positionScreen.y >= - 1 && positionScreen.y <= 1 &&
						 positionScreen.z >= - 1 && positionScreen.z <= 1;

			}

			function pushVertex( x, y, z ) {

				_vertex = getNextVertexInPool();
				_vertex.position.set( x, y, z );

				projectVertex( _vertex );

			}

			function pushNormal( x, y, z ) {

				normals.push( x, y, z );

			}

			function pushColor( r, g, b ) {

				colors.push( r, g, b );

			}

			function pushUv( x, y ) {

				uvs.push( x, y );

			}

			function checkTriangleVisibility( v1, v2, v3 ) {

				if ( v1.visible === true || v2.visible === true || v3.visible === true ) return true;

				_points3[ 0 ] = v1.positionScreen;
				_points3[ 1 ] = v2.positionScreen;
				_points3[ 2 ] = v3.positionScreen;

				return _clipBox.intersectsBox( _boundingBox.setFromPoints( _points3 ) );

			}

			function checkBackfaceCulling( v1, v2, v3 ) {

				return ( ( v3.positionScreen.x - v1.positionScreen.x ) *
					    ( v2.positionScreen.y - v1.positionScreen.y ) -
					    ( v3.positionScreen.y - v1.positionScreen.y ) *
					    ( v2.positionScreen.x - v1.positionScreen.x ) ) < 0;

			}

			function pushLine( a, b ) {

				const v1 = _vertexPool[ a ];
				const v2 = _vertexPool[ b ];

				// Clip

				v1.positionScreen.copy( v1.position ).applyMatrix4( _modelViewProjectionMatrix );
				v2.positionScreen.copy( v2.position ).applyMatrix4( _modelViewProjectionMatrix );

				if ( clipLine( v1.positionScreen, v2.positionScreen ) === true ) {

					// Perform the perspective divide
					v1.positionScreen.multiplyScalar( 1 / v1.positionScreen.w );
					v2.positionScreen.multiplyScalar( 1 / v2.positionScreen.w );

					_line = getNextLineInPool();
					_line.id = object.id;
					_line.v1.copy( v1 );
					_line.v2.copy( v2 );
					_line.z = Math.max( v1.positionScreen.z, v2.positionScreen.z );
					_line.renderOrder = object.renderOrder;

					_line.material = object.material;

					if ( object.material.vertexColors ) {

						_line.vertexColors[ 0 ].fromArray( colors, a * 3 );
						_line.vertexColors[ 1 ].fromArray( colors, b * 3 );

					}

					_renderData.elements.push( _line );

				}

			}

			function pushTriangle( a, b, c, material ) {

				const v1 = _vertexPool[ a ];
				const v2 = _vertexPool[ b ];
				const v3 = _vertexPool[ c ];

				if ( checkTriangleVisibility( v1, v2, v3 ) === false ) return;

				if ( material.side === DoubleSide || checkBackfaceCulling( v1, v2, v3 ) === true ) {

					_face = getNextFaceInPool();

					_face.id = object.id;
					_face.v1.copy( v1 );
					_face.v2.copy( v2 );
					_face.v3.copy( v3 );
					_face.z = ( v1.positionScreen.z + v2.positionScreen.z + v3.positionScreen.z ) / 3;
					_face.renderOrder = object.renderOrder;

					// face normal
					_vector3.subVectors( v3.position, v2.position );
					_vector4.subVectors( v1.position, v2.position );
					_vector3.cross( _vector4 );
					_face.normalModel.copy( _vector3 );
					_face.normalModel.applyMatrix3( normalMatrix ).normalize();

					for ( let i = 0; i < 3; i ++ ) {

						const normal = _face.vertexNormalsModel[ i ];
						normal.fromArray( normals, arguments[ i ] * 3 );
						normal.applyMatrix3( normalMatrix ).normalize();

						const uv = _face.uvs[ i ];
						uv.fromArray( uvs, arguments[ i ] * 2 );

					}

					_face.vertexNormalsLength = 3;

					_face.material = material;

					if ( material.vertexColors ) {

						_face.color.fromArray( colors, a * 3 );

					}

					_renderData.elements.push( _face );

				}

			}

			return {
				setObject: setObject,
				projectVertex: projectVertex,
				checkTriangleVisibility: checkTriangleVisibility,
				checkBackfaceCulling: checkBackfaceCulling,
				pushVertex: pushVertex,
				pushNormal: pushNormal,
				pushColor: pushColor,
				pushUv: pushUv,
				pushLine: pushLine,
				pushTriangle: pushTriangle
			};

		}

		const renderList = new RenderList();

		function projectObject( object ) {

			if ( object.visible === false ) return;

			if ( object.isLight ) {

				_renderData.lights.push( object );

			} else if ( object.isMesh || object.isLine || object.isPoints ) {

				if ( object.material.visible === false ) return;
				if ( object.frustumCulled === true && _frustum.intersectsObject( object ) === false ) return;

				addObject( object );

			} else if ( object.isSprite ) {

				if ( object.material.visible === false ) return;
				if ( object.frustumCulled === true && _frustum.intersectsSprite( object ) === false ) return;

				addObject( object );

			}

			const children = object.children;

			for ( let i = 0, l = children.length; i < l; i ++ ) {

				projectObject( children[ i ] );

			}

		}

		function addObject( object ) {

			_object = getNextObjectInPool();
			_object.id = object.id;
			_object.object = object;

			_vector3.setFromMatrixPosition( object.matrixWorld );
			_vector3.applyMatrix4( _viewProjectionMatrix );
			_object.z = _vector3.z;
			_object.renderOrder = object.renderOrder;

			_renderData.objects.push( _object );

		}

		this.projectScene = function ( scene, camera, sortObjects, sortElements ) {

			_faceCount = 0;
			_lineCount = 0;
			_spriteCount = 0;

			_renderData.elements.length = 0;

			if ( scene.matrixWorldAutoUpdate === true ) scene.updateMatrixWorld();
			if ( camera.parent === null && camera.matrixWorldAutoUpdate === true ) camera.updateMatrixWorld();

			_viewMatrix.copy( camera.matrixWorldInverse );
			_viewProjectionMatrix.multiplyMatrices( camera.projectionMatrix, _viewMatrix );

			_frustum.setFromProjectionMatrix( _viewProjectionMatrix );

			//

			_objectCount = 0;

			_renderData.objects.length = 0;
			_renderData.lights.length = 0;

			projectObject( scene );

			if ( sortObjects === true ) {

				_renderData.objects.sort( painterSort );

			}

			//

			const objects = _renderData.objects;

			for ( let o = 0, ol = objects.length; o < ol; o ++ ) {

				const object = objects[ o ].object;
				const geometry = object.geometry;

				renderList.setObject( object );

				_modelMatrix = object.matrixWorld;

				_vertexCount = 0;

				if ( object.isMesh ) {

					let material = object.material;

					const isMultiMaterial = Array.isArray( material );

					const attributes = geometry.attributes;
					const groups = geometry.groups;

					if ( attributes.position === undefined ) continue;

					const positions = attributes.position.array;

					for ( let i = 0, l = positions.length; i < l; i += 3 ) {

						let x = positions[ i ];
						let y = positions[ i + 1 ];
						let z = positions[ i + 2 ];

						const morphTargets = geometry.morphAttributes.position;

						if ( morphTargets !== undefined ) {

							const morphTargetsRelative = geometry.morphTargetsRelative;
							const morphInfluences = object.morphTargetInfluences;

							for ( let t = 0, tl = morphTargets.length; t < tl; t ++ ) {

								const influence = morphInfluences[ t ];

								if ( influence === 0 ) continue;

								const target = morphTargets[ t ];

								if ( morphTargetsRelative ) {

									x += target.getX( i / 3 ) * influence;
									y += target.getY( i / 3 ) * influence;
									z += target.getZ( i / 3 ) * influence;

								} else {

									x += ( target.getX( i / 3 ) - positions[ i ] ) * influence;
									y += ( target.getY( i / 3 ) - positions[ i + 1 ] ) * influence;
									z += ( target.getZ( i / 3 ) - positions[ i + 2 ] ) * influence;

								}

							}

						}

						renderList.pushVertex( x, y, z );

					}

					if ( attributes.normal !== undefined ) {

						const normals = attributes.normal.array;

						for ( let i = 0, l = normals.length; i < l; i += 3 ) {

							renderList.pushNormal( normals[ i ], normals[ i + 1 ], normals[ i + 2 ] );

						}

					}

					if ( attributes.color !== undefined ) {

						const colors = attributes.color.array;

						for ( let i = 0, l = colors.length; i < l; i += 3 ) {

							renderList.pushColor( colors[ i ], colors[ i + 1 ], colors[ i + 2 ] );

						}

					}

					if ( attributes.uv !== undefined ) {

						const uvs = attributes.uv.array;

						for ( let i = 0, l = uvs.length; i < l; i += 2 ) {

							renderList.pushUv( uvs[ i ], uvs[ i + 1 ] );

						}

					}

					if ( geometry.index !== null ) {

						const indices = geometry.index.array;

						if ( groups.length > 0 ) {

							for ( let g = 0; g < groups.length; g ++ ) {

								const group = groups[ g ];

								material = isMultiMaterial === true
									 ? object.material[ group.materialIndex ]
									 : object.material;

								if ( material === undefined ) continue;

								for ( let i = group.start, l = group.start + group.count; i < l; i += 3 ) {

									renderList.pushTriangle( indices[ i ], indices[ i + 1 ], indices[ i + 2 ], material );

								}

							}

						} else {

							for ( let i = 0, l = indices.length; i < l; i += 3 ) {

								renderList.pushTriangle( indices[ i ], indices[ i + 1 ], indices[ i + 2 ], material );

							}

						}

					} else {

						if ( groups.length > 0 ) {

							for ( let g = 0; g < groups.length; g ++ ) {

								const group = groups[ g ];

								material = isMultiMaterial === true
									 ? object.material[ group.materialIndex ]
									 : object.material;

								if ( material === undefined ) continue;

								for ( let i = group.start, l = group.start + group.count; i < l; i += 3 ) {

									renderList.pushTriangle( i, i + 1, i + 2, material );

								}

							}

						} else {

							for ( let i = 0, l = positions.length / 3; i < l; i += 3 ) {

								renderList.pushTriangle( i, i + 1, i + 2, material );

							}

						}

					}

				} else if ( object.isLine ) {

					_modelViewProjectionMatrix.multiplyMatrices( _viewProjectionMatrix, _modelMatrix );

					const attributes = geometry.attributes;

					if ( attributes.position !== undefined ) {

						const positions = attributes.position.array;

						for ( let i = 0, l = positions.length; i < l; i += 3 ) {

							renderList.pushVertex( positions[ i ], positions[ i + 1 ], positions[ i + 2 ] );

						}

						if ( attributes.color !== undefined ) {

							const colors = attributes.color.array;

							for ( let i = 0, l = colors.length; i < l; i += 3 ) {

								renderList.pushColor( colors[ i ], colors[ i + 1 ], colors[ i + 2 ] );

							}

						}

						if ( geometry.index !== null ) {

							const indices = geometry.index.array;

							for ( let i = 0, l = indices.length; i < l; i += 2 ) {

								renderList.pushLine( indices[ i ], indices[ i + 1 ] );

							}

						} else {

							const step = object.isLineSegments ? 2 : 1;

							for ( let i = 0, l = ( positions.length / 3 ) - 1; i < l; i += step ) {

								renderList.pushLine( i, i + 1 );

							}

						}

					}

				} else if ( object.isPoints ) {

					_modelViewProjectionMatrix.multiplyMatrices( _viewProjectionMatrix, _modelMatrix );

					const attributes = geometry.attributes;

					if ( attributes.position !== undefined ) {

						const positions = attributes.position.array;

						for ( let i = 0, l = positions.length; i < l; i += 3 ) {

							_vector4.set( positions[ i ], positions[ i + 1 ], positions[ i + 2 ], 1 );
							_vector4.applyMatrix4( _modelViewProjectionMatrix );

							pushPoint( _vector4, object, camera );

						}

					}

				} else if ( object.isSprite ) {

					object.modelViewMatrix.multiplyMatrices( camera.matrixWorldInverse, object.matrixWorld );
					_vector4.set( _modelMatrix.elements[ 12 ], _modelMatrix.elements[ 13 ], _modelMatrix.elements[ 14 ], 1 );
					_vector4.applyMatrix4( _viewProjectionMatrix );

					pushPoint( _vector4, object, camera );

				}

			}

			if ( sortElements === true ) {

				_renderData.elements.sort( painterSort );

			}

			return _renderData;

		};

		function pushPoint( _vector4, object, camera ) {

			const invW = 1 / _vector4.w;

			_vector4.z *= invW;

			if ( _vector4.z >= - 1 && _vector4.z <= 1 ) {

				_sprite = getNextSpriteInPool();
				_sprite.id = object.id;
				_sprite.x = _vector4.x * invW;
				_sprite.y = _vector4.y * invW;
				_sprite.z = _vector4.z;
				_sprite.renderOrder = object.renderOrder;
				_sprite.object = object;

				_sprite.rotation = object.rotation;

				_sprite.scale.x = object.scale.x * Math.abs( _sprite.x - ( _vector4.x + camera.projectionMatrix.elements[ 0 ] ) / ( _vector4.w + camera.projectionMatrix.elements[ 12 ] ) );
				_sprite.scale.y = object.scale.y * Math.abs( _sprite.y - ( _vector4.y + camera.projectionMatrix.elements[ 5 ] ) / ( _vector4.w + camera.projectionMatrix.elements[ 13 ] ) );

				_sprite.material = object.material;

				_renderData.elements.push( _sprite );

			}

		}

		// Pools

		function getNextObjectInPool() {

			if ( _objectCount === _objectPoolLength ) {

				const object = new RenderableObject();
				_objectPool.push( object );
				_objectPoolLength ++;
				_objectCount ++;
				return object;

			}

			return _objectPool[ _objectCount ++ ];

		}

		function getNextVertexInPool() {

			if ( _vertexCount === _vertexPoolLength ) {

				const vertex = new RenderableVertex();
				_vertexPool.push( vertex );
				_vertexPoolLength ++;
				_vertexCount ++;
				return vertex;

			}

			return _vertexPool[ _vertexCount ++ ];

		}

		function getNextFaceInPool() {

			if ( _faceCount === _facePoolLength ) {

				const face = new RenderableFace();
				_facePool.push( face );
				_facePoolLength ++;
				_faceCount ++;
				return face;

			}

			return _facePool[ _faceCount ++ ];


		}

		function getNextLineInPool() {

			if ( _lineCount === _linePoolLength ) {

				const line = new RenderableLine();
				_linePool.push( line );
				_linePoolLength ++;
				_lineCount ++;
				return line;

			}

			return _linePool[ _lineCount ++ ];

		}

		function getNextSpriteInPool() {

			if ( _spriteCount === _spritePoolLength ) {

				const sprite = new RenderableSprite();
				_spritePool.push( sprite );
				_spritePoolLength ++;
				_spriteCount ++;
				return sprite;

			}

			return _spritePool[ _spriteCount ++ ];

		}

		//

		function painterSort( a, b ) {

			if ( a.renderOrder !== b.renderOrder ) {

				return a.renderOrder - b.renderOrder;

			} else if ( a.z !== b.z ) {

				return b.z - a.z;

			} else if ( a.id !== b.id ) {

				return a.id - b.id;

			} else {

				return 0;

			}

		}

		function clipLine( s1, s2 ) {

			let alpha1 = 0, alpha2 = 1;

			// Calculate the boundary coordinate of each vertex for the near and far clip planes,
			// Z = -1 and Z = +1, respectively.

			const bc1near = s1.z + s1.w,
				bc2near = s2.z + s2.w,
				bc1far = - s1.z + s1.w,
				bc2far = - s2.z + s2.w;

			if ( bc1near >= 0 && bc2near >= 0 && bc1far >= 0 && bc2far >= 0 ) {

				// Both vertices lie entirely within all clip planes.
				return true;

			} else if ( ( bc1near < 0 && bc2near < 0 ) || ( bc1far < 0 && bc2far < 0 ) ) {

				// Both vertices lie entirely outside one of the clip planes.
				return false;

			} else {

				// The line segment spans at least one clip plane.

				if ( bc1near < 0 ) {

					// v1 lies outside the near plane, v2 inside
					alpha1 = Math.max( alpha1, bc1near / ( bc1near - bc2near ) );

				} else if ( bc2near < 0 ) {

					// v2 lies outside the near plane, v1 inside
					alpha2 = Math.min( alpha2, bc1near / ( bc1near - bc2near ) );

				}

				if ( bc1far < 0 ) {

					// v1 lies outside the far plane, v2 inside
					alpha1 = Math.max( alpha1, bc1far / ( bc1far - bc2far ) );

				} else if ( bc2far < 0 ) {

					// v2 lies outside the far plane, v2 inside
					alpha2 = Math.min( alpha2, bc1far / ( bc1far - bc2far ) );

				}

				if ( alpha2 < alpha1 ) {

					// The line segment spans two boundaries, but is outside both of them.
					// (This can't happen when we're only clipping against just near/far but good
					//  to leave the check here for future usage if other clip planes are added.)
					return false;

				} else {

					// Update the s1 and s2 vertices to match the clipped line segment.
					s1.lerp( s2, alpha1 );
					s2.lerp( s1, 1 - alpha2 );

					return true;

				}

			}

		}

	}

}

class SVGRenderer {

	constructor() {

		let _renderData, _elements, _lights,
			_svgWidth, _svgHeight, _svgWidthHalf, _svgHeightHalf,

			_v1, _v2, _v3,

			_svgNode,
			_pathCount = 0,

			_precision = null,
			_quality = 1,

			_currentPath, _currentStyle;

		const _this = this,
			_clipBox = new Box2(),
			_elemBox = new Box2(),

			_color = new Color(),
			_diffuseColor = new Color(),
			_ambientLight = new Color(),
			_directionalLights = new Color(),
			_pointLights = new Color(),
			_clearColor = new Color(),

			_vector3 = new Vector3(), // Needed for PointLight
			_centroid = new Vector3(),
			_normal = new Vector3(),
			_normalViewMatrix = new Matrix3(),

			_viewMatrix = new Matrix4(),
			_viewProjectionMatrix = new Matrix4(),

			_svgPathPool = [],

			_projector = new Projector(),
			_svg = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' );

		this.domElement = _svg;

		this.autoClear = true;
		this.sortObjects = true;
		this.sortElements = true;

		this.overdraw = 0.5;

		this.outputColorSpace = SRGBColorSpace;

		this.info = {

			render: {

				vertices: 0,
				faces: 0

			}

		};

		this.setQuality = function ( quality ) {

			switch ( quality ) {

				case 'high': _quality = 1; break;
				case 'low': _quality = 0; break;

			}

		};

		this.setClearColor = function ( color ) {

			_clearColor.set( color );

		};

		this.setPixelRatio = function () {};

		this.setSize = function ( width, height ) {

			_svgWidth = width; _svgHeight = height;
			_svgWidthHalf = _svgWidth / 2; _svgHeightHalf = _svgHeight / 2;

			_svg.setAttribute( 'viewBox', ( - _svgWidthHalf ) + ' ' + ( - _svgHeightHalf ) + ' ' + _svgWidth + ' ' + _svgHeight );
			_svg.setAttribute( 'width', _svgWidth );
			_svg.setAttribute( 'height', _svgHeight );

			_clipBox.min.set( - _svgWidthHalf, - _svgHeightHalf );
			_clipBox.max.set( _svgWidthHalf, _svgHeightHalf );

		};

		this.getSize = function () {

			return {
				width: _svgWidth,
				height: _svgHeight
			};

		};

		this.setPrecision = function ( precision ) {

			_precision = precision;

		};

		function removeChildNodes() {

			_pathCount = 0;

			while ( _svg.childNodes.length > 0 ) {

				_svg.removeChild( _svg.childNodes[ 0 ] );

			}

		}

		function convert( c ) {

			return _precision !== null ? c.toFixed( _precision ) : c;

		}

		this.clear = function () {

			removeChildNodes();
			_svg.style.backgroundColor = _clearColor.getStyle( _this.outputColorSpace );

		};

		this.render = function ( scene, camera ) {

			if ( camera instanceof Camera === false ) {

				console.error( 'THREE.SVGRenderer.render: camera is not an instance of Camera.' );
				return;

			}

			const background = scene.background;

			if ( background && background.isColor ) {

				removeChildNodes();
				_svg.style.backgroundColor = background.getStyle( _this.outputColorSpace );

			} else if ( this.autoClear === true ) {

				this.clear();

			}

			_this.info.render.vertices = 0;
			_this.info.render.faces = 0;

			_viewMatrix.copy( camera.matrixWorldInverse );
			_viewProjectionMatrix.multiplyMatrices( camera.projectionMatrix, _viewMatrix );

			_renderData = _projector.projectScene( scene, camera, this.sortObjects, this.sortElements );
			_elements = _renderData.elements;
			_lights = _renderData.lights;

			_normalViewMatrix.getNormalMatrix( camera.matrixWorldInverse );

			calculateLights( _lights );

			 // reset accumulated path

			_currentPath = '';
			_currentStyle = '';

			for ( let e = 0, el = _elements.length; e < el; e ++ ) {

				const element = _elements[ e ];
				const material = element.material;

				if ( material === undefined || material.opacity === 0 ) continue;

				_elemBox.makeEmpty();

				if ( element instanceof RenderableSprite ) {

					_v1 = element;
					_v1.x *= _svgWidthHalf; _v1.y *= - _svgHeightHalf;

					renderSprite( _v1, element, material );

				} else if ( element instanceof RenderableLine ) {

					_v1 = element.v1; _v2 = element.v2;

					_v1.positionScreen.x *= _svgWidthHalf; _v1.positionScreen.y *= - _svgHeightHalf;
					_v2.positionScreen.x *= _svgWidthHalf; _v2.positionScreen.y *= - _svgHeightHalf;

					_elemBox.setFromPoints( [ _v1.positionScreen, _v2.positionScreen ] );

					if ( _clipBox.intersectsBox( _elemBox ) === true ) {

						renderLine( _v1, _v2, material );

					}

				} else if ( element instanceof RenderableFace ) {

					_v1 = element.v1; _v2 = element.v2; _v3 = element.v3;

					if ( _v1.positionScreen.z < - 1 || _v1.positionScreen.z > 1 ) continue;
					if ( _v2.positionScreen.z < - 1 || _v2.positionScreen.z > 1 ) continue;
					if ( _v3.positionScreen.z < - 1 || _v3.positionScreen.z > 1 ) continue;

					_v1.positionScreen.x *= _svgWidthHalf; _v1.positionScreen.y *= - _svgHeightHalf;
					_v2.positionScreen.x *= _svgWidthHalf; _v2.positionScreen.y *= - _svgHeightHalf;
					_v3.positionScreen.x *= _svgWidthHalf; _v3.positionScreen.y *= - _svgHeightHalf;

					if ( this.overdraw > 0 ) {

						expand( _v1.positionScreen, _v2.positionScreen, this.overdraw );
						expand( _v2.positionScreen, _v3.positionScreen, this.overdraw );
						expand( _v3.positionScreen, _v1.positionScreen, this.overdraw );

					}

					_elemBox.setFromPoints( [
						_v1.positionScreen,
						_v2.positionScreen,
						_v3.positionScreen
					] );

					if ( _clipBox.intersectsBox( _elemBox ) === true ) {

						renderFace3( _v1, _v2, _v3, element, material );

					}

				}

			}

			flushPath(); // just to flush last svg:path

			scene.traverseVisible( function ( object ) {

				 if ( object.isSVGObject ) {

					_vector3.setFromMatrixPosition( object.matrixWorld );
					_vector3.applyMatrix4( _viewProjectionMatrix );

					if ( _vector3.z < - 1 || _vector3.z > 1 ) return;

					const x = _vector3.x * _svgWidthHalf;
					const y = - _vector3.y * _svgHeightHalf;

					const node = object.node;
					node.setAttribute( 'transform', 'translate(' + x + ',' + y + ')' );

					_svg.appendChild( node );

				}

			} );

		};

		function calculateLights( lights ) {

			_ambientLight.setRGB( 0, 0, 0 );
			_directionalLights.setRGB( 0, 0, 0 );
			_pointLights.setRGB( 0, 0, 0 );

			for ( let l = 0, ll = lights.length; l < ll; l ++ ) {

				const light = lights[ l ];
				const lightColor = light.color;

				if ( light.isAmbientLight ) {

					_ambientLight.r += lightColor.r;
					_ambientLight.g += lightColor.g;
					_ambientLight.b += lightColor.b;

				} else if ( light.isDirectionalLight ) {

					_directionalLights.r += lightColor.r;
					_directionalLights.g += lightColor.g;
					_directionalLights.b += lightColor.b;

				} else if ( light.isPointLight ) {

					_pointLights.r += lightColor.r;
					_pointLights.g += lightColor.g;
					_pointLights.b += lightColor.b;

				}

			}

		}

		function calculateLight( lights, position, normal, color ) {

			for ( let l = 0, ll = lights.length; l < ll; l ++ ) {

				const light = lights[ l ];
				const lightColor = light.color;

				if ( light.isDirectionalLight ) {

					const lightPosition = _vector3.setFromMatrixPosition( light.matrixWorld ).normalize();

					let amount = normal.dot( lightPosition );

					if ( amount <= 0 ) continue;

					amount *= light.intensity;

					color.r += lightColor.r * amount;
					color.g += lightColor.g * amount;
					color.b += lightColor.b * amount;

				} else if ( light.isPointLight ) {

					const lightPosition = _vector3.setFromMatrixPosition( light.matrixWorld );

					let amount = normal.dot( _vector3.subVectors( lightPosition, position ).normalize() );

					if ( amount <= 0 ) continue;

					amount *= light.distance == 0 ? 1 : 1 - Math.min( position.distanceTo( lightPosition ) / light.distance, 1 );

					if ( amount == 0 ) continue;

					amount *= light.intensity;

					color.r += lightColor.r * amount;
					color.g += lightColor.g * amount;
					color.b += lightColor.b * amount;

				}

			}

		}

		function renderSprite( v1, element, material ) {

			let scaleX = element.scale.x * _svgWidthHalf;
			let scaleY = element.scale.y * _svgHeightHalf;

			if ( material.isPointsMaterial ) {

				scaleX *= material.size;
				scaleY *= material.size;

			}

			const path = 'M' + convert( v1.x - scaleX * 0.5 ) + ',' + convert( v1.y - scaleY * 0.5 ) + 'h' + convert( scaleX ) + 'v' + convert( scaleY ) + 'h' + convert( - scaleX ) + 'z';
			let style = '';

			if ( material.isSpriteMaterial || material.isPointsMaterial ) {

				style = 'fill:' + material.color.getStyle( _this.outputColorSpace ) + ';fill-opacity:' + material.opacity;

			}

			addPath( style, path );

		}

		function renderLine( v1, v2, material ) {

			const path = 'M' + convert( v1.positionScreen.x ) + ',' + convert( v1.positionScreen.y ) + 'L' + convert( v2.positionScreen.x ) + ',' + convert( v2.positionScreen.y );

			if ( material.isLineBasicMaterial ) {

				let style = 'fill:none;stroke:' + material.color.getStyle( _this.outputColorSpace ) + ';stroke-opacity:' + material.opacity + ';stroke-width:' + material.linewidth + ';stroke-linecap:' + material.linecap;

				if ( material.isLineDashedMaterial ) {

					style = style + ';stroke-dasharray:' + material.dashSize + ',' + material.gapSize;

				}

				addPath( style, path );

			}

		}

		function renderFace3( v1, v2, v3, element, material ) {

			_this.info.render.vertices += 3;
			_this.info.render.faces ++;

			const path = 'M' + convert( v1.positionScreen.x ) + ',' + convert( v1.positionScreen.y ) + 'L' + convert( v2.positionScreen.x ) + ',' + convert( v2.positionScreen.y ) + 'L' + convert( v3.positionScreen.x ) + ',' + convert( v3.positionScreen.y ) + 'z';
			let style = '';

			if ( material.isMeshBasicMaterial ) {

				_color.copy( material.color );

				if ( material.vertexColors ) {

					_color.multiply( element.color );

				}

			} else if ( material.isMeshLambertMaterial || material.isMeshPhongMaterial || material.isMeshStandardMaterial ) {

				_diffuseColor.copy( material.color );

				if ( material.vertexColors ) {

					_diffuseColor.multiply( element.color );

				}

				_color.copy( _ambientLight );

				_centroid.copy( v1.positionWorld ).add( v2.positionWorld ).add( v3.positionWorld ).divideScalar( 3 );

				calculateLight( _lights, _centroid, element.normalModel, _color );

				_color.multiply( _diffuseColor ).add( material.emissive );

			} else if ( material.isMeshNormalMaterial ) {

				_normal.copy( element.normalModel ).applyMatrix3( _normalViewMatrix ).normalize();

				_color.setRGB( _normal.x, _normal.y, _normal.z ).multiplyScalar( 0.5 ).addScalar( 0.5 );

			}

			if ( material.wireframe ) {

				style = 'fill:none;stroke:' + _color.getStyle( _this.outputColorSpace ) + ';stroke-opacity:' + material.opacity + ';stroke-width:' + material.wireframeLinewidth + ';stroke-linecap:' + material.wireframeLinecap + ';stroke-linejoin:' + material.wireframeLinejoin;

			} else {

				style = 'fill:' + _color.getStyle( _this.outputColorSpace ) + ';fill-opacity:' + material.opacity;

			}

			addPath( style, path );

		}

		// Hide anti-alias gaps

		function expand( v1, v2, pixels ) {

			let x = v2.x - v1.x, y = v2.y - v1.y;
			const det = x * x + y * y;

			if ( det === 0 ) return;

			const idet = pixels / Math.sqrt( det );

			x *= idet; y *= idet;

			v2.x += x; v2.y += y;
			v1.x -= x; v1.y -= y;

		}

		function addPath( style, path ) {

			if ( _currentStyle === style ) {

				_currentPath += path;

			} else {

				flushPath();

				_currentStyle = style;
				_currentPath = path;

			}

		}

		function flushPath() {

			if ( _currentPath ) {

				_svgNode = getPathNode( _pathCount ++ );
				_svgNode.setAttribute( 'd', _currentPath );
				_svgNode.setAttribute( 'style', _currentStyle );
				_svg.appendChild( _svgNode );

			}

			_currentPath = '';
			_currentStyle = '';

		}

		function getPathNode( id ) {

			if ( _svgPathPool[ id ] == null ) {

				_svgPathPool[ id ] = document.createElementNS( 'http://www.w3.org/2000/svg', 'path' );

				if ( _quality == 0 ) {

					_svgPathPool[ id ].setAttribute( 'shape-rendering', 'crispEdges' ); //optimizeSpeed

				}

				return _svgPathPool[ id ];

			}

			return _svgPathPool[ id ];

		}

	}

}

export { CopyShader, EffectComposer, Font, LuminosityHighPassShader, MaskPass, OrbitControls, Pass, RenderPass, SVGRenderer, ShaderPass, SimplexNoise, TextGeometry, UnrealBloomPass };
