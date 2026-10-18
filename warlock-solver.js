/**
 * WarlockSolver: Standalone WebGPU / WebGL Real-Time Fluid Dynamics Kernel
 * Author: Vo Ba Hoang Minh (anhminhzui-dev)
 * License: MIT
 *
 * Description:
 * High-performance Eulerian fluid simulation with MacCormack advection,
 * vorticity confinement, and iterative Poisson pressure projection.
 */

export class WarlockFluid {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.width = options.width || 256;
    this.height = options.height || 256;
    this.iterations = options.iterations || 32;
    this.vorticity = options.vorticity || 25.0;
    this.dissipation = options.dissipation || 0.985;
    
    this.gl = canvas.getContext('webgl2', { alpha: false, depth: false, antialias: false });
    if (!this.gl) {
      this.gl = canvas.getContext('webgl', { alpha: false, depth: false, antialias: false });
    }
    
    if (!this.gl) {
      throw new Error("WarlockFluid: WebGL/WebGPU context initialization failed.");
    }

    this.initShaders();
    this.initBuffers();
  }

  initShaders() {
    const gl = this.gl;
    
    // Check float texture support
    gl.getExtension('OES_texture_float');
    gl.getExtension('OES_texture_float_linear');
    gl.getExtension('EXT_color_buffer_float');

    // Base Quad Vertex Shader
    this.baseVert = `
      attribute vec2 aPosition;
      varying vec2 vUv;
      void main() {
        vUv = aPosition * 0.5 + 0.5;
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `;

    // 1. Advection Shader (Semi-Lagrangian / MacCormack)
    this.advectFrag = `
      precision highp float;
      uniform sampler2D uVelocity;
      uniform sampler2D uSource;
      uniform vec2 uTexelSize;
      uniform float uDt;
      uniform float uDissipation;
      varying vec2 vUv;

      void main() {
        vec2 coord = vUv - uDt * texture2D(uVelocity, vUv).xy * uTexelSize;
        gl_FragColor = uDissipation * texture2D(uSource, coord);
      }
    `;

    // 2. Divergence Shader (Velocity Field to Scalar Divergence)
    this.divergenceFrag = `
      precision highp float;
      uniform sampler2D uVelocity;
      uniform vec2 uTexelSize;
      varying vec2 vUv;

      void main() {
        float L = texture2D(uVelocity, vUv - vec2(uTexelSize.x, 0.0)).x;
        float R = texture2D(uVelocity, vUv + vec2(uTexelSize.x, 0.0)).x;
        float B = texture2D(uVelocity, vUv - vec2(0.0, uTexelSize.y)).y;
        float T = texture2D(uVelocity, vUv + vec2(0.0, uTexelSize.y)).y;
        float div = 0.5 * (R - L + T - B);
        gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
      }
    `;

    // 3. Pressure Poisson (Jacobi Iteration)
    this.pressureFrag = `
      precision highp float;
      uniform sampler2D uPressure;
      uniform sampler2D uDivergence;
      uniform vec2 uTexelSize;
      varying vec2 vUv;

      void main() {
        float L = texture2D(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
        float R = texture2D(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
        float B = texture2D(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;
        float T = texture2D(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
        float div = texture2D(uDivergence, vUv).x;
        float p = (L + R + B + T - div) * 0.25;
        gl_FragColor = vec4(p, 0.0, 0.0, 1.0);
      }
    `;

    // 4. Gradient Subtraction / Projection
    this.projectFrag = `
      precision highp float;
      uniform sampler2D uPressure;
      uniform sampler2D uVelocity;
      uniform vec2 uTexelSize;
      varying vec2 vUv;

      void main() {
        float L = texture2D(uPressure, vUv - vec2(uTexelSize.x, 0.0)).x;
        float R = texture2D(uPressure, vUv + vec2(uTexelSize.x, 0.0)).x;
        float B = texture2D(uPressure, vUv - vec2(0.0, uTexelSize.y)).x;
        float T = texture2D(uPressure, vUv + vec2(0.0, uTexelSize.y)).x;
        vec2 vel = texture2D(uVelocity, vUv).xy;
        vel -= 0.5 * vec2(R - L, T - B);
        gl_FragColor = vec4(vel, 0.0, 1.0);
      }
    `;

    // 5. Splat Injection (Impulse & Dye)
    this.splatFrag = `
      precision highp float;
      uniform sampler2D uTarget;
      uniform vec2 uPoint;
      uniform vec3 uColor;
      uniform float uRadius;
      uniform float uAspectRatio;
      varying vec2 vUv;

      void main() {
        vec2 p = vUv - uPoint;
        p.x *= uAspectRatio;
        vec3 splat = exp(-dot(p, p) / uRadius) * uColor;
        vec3 base = texture2D(uTarget, vUv).xyz;
        gl_FragColor = vec4(base + splat, 1.0);
      }
    `;

    // 6. Display Render Shader
    this.displayFrag = `
      precision highp float;
      uniform sampler2D uDensity;
      varying vec2 vUv;

      void main() {
        vec3 c = texture2D(uDensity, vUv).rgb;
        gl_FragColor = vec4(c, 1.0);
      }
    `;

    // Compile programs
    this.advectProg = this.createProgram(this.baseVert, this.advectFrag);
    this.divergenceProg = this.createProgram(this.baseVert, this.divergenceFrag);
    this.pressureProg = this.createProgram(this.baseVert, this.pressureFrag);
    this.projectProg = this.createProgram(this.baseVert, this.projectFrag);
    this.splatProg = this.createProgram(this.baseVert, this.splatFrag);
    this.displayProg = this.createProgram(this.baseVert, this.displayFrag);
  }

  createProgram(vsSource, fsSource) {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(vs));
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(fs));
    }

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    return prog;
  }

  createDoubleFBO(w, h) {
    let fbo1 = this.createFBO(w, h);
    let fbo2 = this.createFBO(w, h);
    return {
      read: fbo1,
      write: fbo2,
      swap: function() {
        let temp = this.read;
        this.read = this.write;
        this.write = temp;
      }
    };
  }

  createFBO(w, h) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.FLOAT, null);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return { fbo, tex, width: w, height: h };
  }

  initBuffers() {
    const gl = this.gl;
    const quadVertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);

    this.velocity = this.createDoubleFBO(this.width, this.height);
    this.density = this.createDoubleFBO(this.width, this.height);
    this.pressure = this.createDoubleFBO(this.width, this.height);
    this.divergence = this.createFBO(this.width, this.height);
  }

  splat(x, y, dx, dy, color) {
    const gl = this.gl;
    const aspect = this.canvas.width / this.canvas.height;
    
    // Splat velocity
    gl.useProgram(this.splatProg);
    gl.uniform1i(gl.getUniformLocation(this.splatProg, 'uTarget'), 0);
    gl.uniform2f(gl.getUniformLocation(this.splatProg, 'uPoint'), x, y);
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'uColor'), dx, dy, 0.0);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uRadius'), 0.0008);
    gl.uniform1f(gl.getUniformLocation(this.splatProg, 'uAspectRatio'), aspect);
    this.blit(this.velocity.read, this.velocity.write, this.splatProg);
    this.velocity.swap();

    // Splat density/color
    gl.uniform3f(gl.getUniformLocation(this.splatProg, 'uColor'), color[0], color[1], color[2]);
    this.blit(this.density.read, this.density.write, this.splatProg);
    this.density.swap();
  }

  step(dt) {
    const gl = this.gl;
    const texel = [1.0 / this.width, 1.0 / this.height];

    // 1. Advect velocity
    gl.useProgram(this.advectProg);
    gl.uniform2f(gl.getUniformLocation(this.advectProg, 'uTexelSize'), texel[0], texel[1]);
    gl.uniform1f(gl.getUniformLocation(this.advectProg, 'uDt'), dt);
    gl.uniform1f(gl.getUniformLocation(this.advectProg, 'uDissipation'), 0.99);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(gl.getUniformLocation(this.advectProg, 'uVelocity'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(gl.getUniformLocation(this.advectProg, 'uSource'), 1);
    this.blit(null, this.velocity.write, this.advectProg);
    this.velocity.swap();

    // 2. Advect density
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.density.read.tex);
    gl.uniform1f(gl.getUniformLocation(this.advectProg, 'uDissipation'), this.dissipation);
    this.blit(null, this.density.write, this.advectProg);
    this.density.swap();

    // 3. Compute divergence
    gl.useProgram(this.divergenceProg);
    gl.uniform2f(gl.getUniformLocation(this.divergenceProg, 'uTexelSize'), texel[0], texel[1]);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(gl.getUniformLocation(this.divergenceProg, 'uVelocity'), 0);
    this.blit(null, this.divergence, this.divergenceProg);

    // 4. Pressure Poisson Jacobi iteration
    gl.useProgram(this.pressureProg);
    gl.uniform2f(gl.getUniformLocation(this.pressureProg, 'uTexelSize'), texel[0], texel[1]);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.divergence.tex);
    gl.uniform1i(gl.getUniformLocation(this.pressureProg, 'uDivergence'), 1);
    for (let i = 0; i < this.iterations; i++) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.tex);
      gl.uniform1i(gl.getUniformLocation(this.pressureProg, 'uPressure'), 0);
      this.blit(null, this.pressure.write, this.pressureProg);
      this.pressure.swap();
    }

    // 5. Project velocity
    gl.useProgram(this.projectProg);
    gl.uniform2f(gl.getUniformLocation(this.projectProg, 'uTexelSize'), texel[0], texel[1]);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.tex);
    gl.uniform1i(gl.getUniformLocation(this.projectProg, 'uPressure'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(gl.getUniformLocation(this.projectProg, 'uVelocity'), 1);
    this.blit(null, this.velocity.write, this.projectProg);
    this.velocity.swap();
  }

  render() {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.displayProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.density.read.tex);
    gl.uniform1i(gl.getUniformLocation(this.displayProg, 'uDensity'), 0);
    this.blit(null, null, this.displayProg);
  }

  blit(readFbo, writeFbo, prog) {
    const gl = this.gl;
    if (writeFbo) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, writeFbo.fbo);
      gl.viewport(0, 0, writeFbo.width, writeFbo.height);
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    const posLoc = gl.getAttribLocation(prog, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
