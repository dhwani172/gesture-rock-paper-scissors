let config = { burst_frames: 11, window_ms: 450, lock_at_count: 2 };
let stream;
let rounds = 0, wins = 0, losses = 0, ties = 0;

const els = {
  cam: document.getElementById('cam'),
  canvas: document.getElementById('snap'),
  start: document.getElementById('startBtn'),
  status: document.getElementById('status'),
  countdown: document.getElementById('countdown'),
  diff: document.getElementById('difficulty'),
  rRounds: document.getElementById('rounds'),
  rWins: document.getElementById('wins'),
  rLosses: document.getElementById('losses'),
  rTies: document.getElementById('ties'),
  rPlayer: document.getElementById('playerMove'),
  rAI: document.getElementById('aiMove'),
  rRes: document.getElementById('result'),
  linkHow: document.getElementById('linkHow'),
  howModal: document.getElementById('howModal'),
};

async function setup() {
  const cfgRes = await fetch('/api/config');
  if (cfgRes.ok) {
    const c = await cfgRes.json();
    config = c;
    els.diff.value = c.default_difficulty || 'adaptive_freq';
  }
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 }, audio: false });
  els.cam.srcObject = stream;
  // no live hand-tracking overlay in pastel version
}

function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }

function captureJPEG(scale=1.0) {
  const v = els.cam;
  const c = els.canvas;
  const w = Math.max(1, Math.floor(v.videoWidth * scale));
  const h = Math.max(1, Math.floor(v.videoHeight * scale));
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.save();
  // mirror to match the OpenCV desktop view
  g.translate(c.width, 0); g.scale(-1, 1);
  g.drawImage(v, 0, 0, c.width, c.height);
  g.restore();
  return c.toDataURL('image/jpeg', 0.7);
}

async function playRound() {
  els.start.classList.add('pulse');
  setTimeout(() => els.start.classList.remove('pulse'), 280);

  // 3..2..1.. Shoot!
  const seq = ['3','2','1','Shoot!'];
  for (let i = 0; i < seq.length; i++) {
    els.countdown.textContent = seq[i];
    await sleep(300);
  }
  els.countdown.textContent = '';

  // Capture burst
  const images = [];
  const target = config.burst_frames || 11;
  const per = Math.max(10, Math.floor((config.window_ms || 450) / target));
  for (let i = 0; i < target; i++) {
    images.push(captureJPEG());
    await sleep(per);
  }

  // Send to server
  els.status.textContent = 'Thinking…';
  const resp = await fetch('/api/round', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ images, difficulty: els.diff.value }),
  });

  if (!resp.ok) {
    els.status.textContent = 'Server error';
    return;
  }
  const r = await resp.json();
  els.status.textContent = '';

  // Update UI
  els.rPlayer.textContent = r.player;
  els.rAI.textContent = r.ai;
  els.rRes.textContent = r.result;
  rounds += 1; els.rRounds.textContent = rounds;
  if (r.result === 'win') { wins += 1; els.rWins.textContent = wins; }
  else if (r.result === 'lose') { losses += 1; els.rLosses.textContent = losses; }
  else if (r.result === 'tie') { ties += 1; els.rTies.textContent = ties; }
}


els.start.addEventListener('click', playRound);
window.addEventListener('keydown', (e) => { if (e.code === 'Space'){ e.preventDefault(); playRound(); }});

window.addEventListener('load', setup);

// ---- Hero FX with anime.js: swirling particle ring and subtle grid drift ----
let hero = { started:false, canvas:null, ctx:null, parts:[], pulse:0.0, t:0 };

function initHeroFX(){
  const c = document.getElementById('heroFx');
  if (!c) return;
  hero.canvas = c; hero.ctx = c.getContext('2d');
  const resize = () => { c.width = c.clientWidth; c.height = c.clientHeight; };
  window.addEventListener('resize', resize); resize();
  // build particles around a ring
  const N = 180; hero.parts = [];
  for (let i=0;i<N;i++){
    const a = (i/N)*Math.PI*2;
    hero.parts.push({ a, r: 140 + Math.random()*40, s: 0.4 + Math.random()*0.8 });
  }
  // anime timeline to gently pulse the ring
  anime({
    targets: hero, pulse: 1.0, duration: 2200, direction: 'alternate', easing: 'easeInOutSine', loop: true
  });
  // subtle grid drift via CSS vars
  const gridObj = { x: 0, y: 0 };
  anime({ targets: gridObj, x: 60, y: 20, duration: 12000, direction:'alternate', easing: 'linear', loop:true,
    update: () => { document.documentElement.style.setProperty('--grid-x', gridObj.x+'px'); document.documentElement.style.setProperty('--grid-y', gridObj.y+'px'); }
  });
  hero.started = true; requestAnimationFrame(drawHero);
}

function drawHero(){
  if (!hero.started) return;
  const ctx = hero.ctx, c = hero.canvas; if (!ctx || !c) return;
  hero.t += 0.016;
  ctx.clearRect(0,0,c.width,c.height);
  const cx = c.width*0.5, cy = Math.min(c.height*0.68, c.height*0.72);
  // central warm glow
  const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, Math.min(c.width, c.height)*0.45);
  grad.addColorStop(0, 'rgba(255,77,46,0.30)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad; ctx.fillRect(0,0,c.width,c.height);
  // ring particles
  for (const p of hero.parts){
    const th = p.a + hero.t * p.s * 0.6;
    const amp = 1.0 + 0.08 * Math.sin(hero.t*2.0 + p.a*3.0) + 0.10*hero.pulse;
    const r = p.r * amp;
    const x = cx + Math.cos(th)*r;
    const y = cy + Math.sin(th)*r*0.55; // ellipse look
    const dot = ctx.createRadialGradient(x-1, y-1, 0, x, y, 8);
    dot.addColorStop(0, 'rgba(255,255,255,0.9)');
    dot.addColorStop(0.5, 'rgba(255,77,46,0.85)');
    dot.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = dot; ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fill();
  }
  requestAnimationFrame(drawHero);
}

// kick off hero FX when DOM ready (after anime.js loads)
document.addEventListener('DOMContentLoaded', () => {
  if (window.anime) {
    initHeroFX();
    // entrance timeline for nav + hero copy + actions
    const tl = anime.timeline({ easing: 'easeOutQuad', duration: 650 });
    tl.add({ targets: '.nav', opacity: [0,1], translateY: [-12, 0] })
      .add({ targets: '.hero-title', opacity: [0,1], translateY: [12,0] }, '-=400')
      .add({ targets: '.hero-subtitle', opacity: [0,1], translateY: [12,0] }, '-=450')
      .add({ targets: '.actions', opacity: [0,1], translateY: [12,0] }, '-=430');

    // remove parallax in pastel version (no hero canvas)
  }
});

// Modal logic
function openModal(el){ el.setAttribute('aria-hidden','false'); }
function closeModal(el){ el.setAttribute('aria-hidden','true'); }
document.addEventListener('click', (e) => {
  const t = e.target;
  if (t.id === 'linkHow') { e.preventDefault(); openModal(els.howModal); }
  if (t.dataset && t.dataset.close === 'modal') { closeModal(els.howModal); }
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(els.howModal); });
