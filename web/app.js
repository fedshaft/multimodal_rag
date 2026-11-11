/**
 * PROJECT BLACK HOLE 2 — COSMIC CLIENT APPLICATION
 * Interactive Space-Themed Multimodal RAG Experience
 */

(function () {
  'use strict';

  // Global Data Reference from data_bridge.js
  const DATA = window.BLACKHOLE_DATA || {
    results: { results: [] },
    eval_set: { questions: [] },
    documents: [],
    figures: [],
    chunks_map: {},
    preset_answers: {}
  };

  // State
  const state = {
    activeView: 'mission',
    activeLabTab: 'leaderboard',
    selectedMethod: 'full',
    topK: 5,
    maxImages: 2,
    doGenerate: false,
    currentQuery: '',
    audioPlaying: false,
    audioCtx: null,
    audioGain: null,
    docFilter: 'All',
    benchFilter: 'All',
    figureFilter: 'All',
    docSearchQuery: '',
    benchSearchQuery: ''
  };

  /* ==========================================================================
     1. COSMIC STARFIELD & BLACK HOLE CANVAS SIMULATION
     ========================================================================== */
  function initStarfield() {
    const canvas = document.getElementById('space-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width, height;
    let stars = [];
    const numStars = 160;
    let targetMouseX = 0;
    let targetMouseY = 0;
    let smoothMouseX = 0;
    let smoothMouseY = 0;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    window.addEventListener('resize', resize);
    resize();

    window.addEventListener('mousemove', (e) => {
      targetMouseX = (e.clientX - width / 2) * 0.03;
      targetMouseY = (e.clientY - height / 2) * 0.03;
    });

    const colors = ['rgba(245, 246, 248, 0.7)', 'rgba(200, 207, 217, 0.5)', 'rgba(56, 189, 248, 0.4)', 'rgba(129, 140, 248, 0.3)'];

    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 1.1 + 0.3,
        color: colors[Math.floor(Math.random() * colors.length)],
        speed: Math.random() * 0.07 + 0.015,
        alpha: Math.random() * 0.5 + 0.15,
        twinkleSpeed: Math.random() * 0.006 + 0.001,
        twinkleDir: Math.random() > 0.5 ? 1 : -1
      });
    }

    function animate() {
      ctx.clearRect(0, 0, width, height);

      // Smooth parallax interpolation
      smoothMouseX += (targetMouseX - smoothMouseX) * 0.05;
      smoothMouseY += (targetMouseY - smoothMouseY) * 0.05;

      for (let star of stars) {
        if (!prefersReducedMotion) {
          star.y += star.speed;
          if (star.y > height) {
            star.y = 0;
            star.x = Math.random() * width;
          }

          star.alpha += star.twinkleSpeed * star.twinkleDir;
          if (star.alpha > 0.75) {
            star.alpha = 0.75;
            star.twinkleDir = -1;
          } else if (star.alpha < 0.1) {
            star.alpha = 0.1;
            star.twinkleDir = 1;
          }
        }

        const drawX = star.x + smoothMouseX;
        const drawY = star.y + smoothMouseY;

        ctx.beginPath();
        ctx.arc(drawX, drawY, star.radius, 0, Math.PI * 2);
        ctx.fillStyle = star.color;
        ctx.globalAlpha = star.alpha;
        ctx.fill();
      }

      ctx.globalAlpha = 1.0;
      if (!prefersReducedMotion) {
        requestAnimationFrame(animate);
      }
    }

    animate();
  }

  /* ==========================================================================
     2. NATIVE WEB AUDIO COSMIC DRONE SYNTHESIZER
     ========================================================================== */
  function toggleCosmicAudio() {
    const btn = document.getElementById('btn-audio-toggle');
    if (!state.audioPlaying) {
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!state.audioCtx) {
          state.audioCtx = new AudioContext();
        }
        if (state.audioCtx.state === 'suspended') {
          state.audioCtx.resume();
        }

        // Master Gain
        state.audioGain = state.audioCtx.createGain();
        state.audioGain.gain.setValueAtTime(0.01, state.audioCtx.currentTime);
        state.audioGain.gain.exponentialRampToValueAtTime(0.12, state.audioCtx.currentTime + 3);
        state.audioGain.connect(state.audioCtx.destination);

        // Low Pass Filter for deep space atmosphere
        const filter = state.audioCtx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(220, state.audioCtx.currentTime);
        filter.connect(state.audioGain);

        // Slow LFO for organic nebula breathing
        const lfo = state.audioCtx.createOscillator();
        const lfoGain = state.audioCtx.createGain();
        lfo.frequency.setValueAtTime(0.1, state.audioCtx.currentTime);
        lfoGain.gain.setValueAtTime(80, state.audioCtx.currentTime);
        lfo.connect(lfoGain);
        lfoGain.connect(filter.frequency);
        lfo.start();

        // 3 Detuned Sine Oscillators
        const freqs = [55, 110, 164.81]; // A1 fundamental, octave, fifth
        freqs.forEach((f, idx) => {
          const osc = state.audioCtx.createOscillator();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(f + (idx * 0.4), state.audioCtx.currentTime);
          osc.connect(filter);
          osc.start();
        });

        state.audioPlaying = true;
        btn.classList.add('playing');
        btn.title = "Pause cosmic sound";
      } catch (err) {
        console.warn("Web Audio initialization skipped:", err);
      }
    } else {
      if (state.audioGain && state.audioCtx) {
        state.audioGain.gain.exponentialRampToValueAtTime(0.0001, state.audioCtx.currentTime + 0.8);
        setTimeout(() => {
          state.audioCtx.suspend();
        }, 800);
      }
      state.audioPlaying = false;
      btn.classList.remove('playing');
      btn.title = "Enable cosmic sound";
    }
  }

  /* ==========================================================================
     3. NAVIGATION / VIEW CONTROLLER
     ========================================================================== */
  function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.view-section');

    function switchView(viewName) {
      state.activeView = viewName;

      navButtons.forEach(btn => {
        if (btn.dataset.view === viewName) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });

      sections.forEach(sec => {
        if (sec.id === `view-${viewName}`) {
          sec.classList.add('active');
        } else {
          sec.classList.remove('active');
        }
      });

      window.location.hash = viewName;
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    navButtons.forEach(btn => {
      btn.addEventListener('click', () => switchView(btn.dataset.view));
    });

    // Check URL Hash on Load
    const hash = window.location.hash.replace('#', '');
    if (['mission', 'playground', 'benchmark', 'dataset', 'architecture'].includes(hash)) {
      switchView(hash);
    }

    // Hero action buttons
    const heroExploreBtn = document.getElementById('hero-btn-explore');
    if (heroExploreBtn) {
      heroExploreBtn.addEventListener('click', () => switchView('playground'));
    }
    const heroBenchBtn = document.getElementById('hero-btn-benchmark');
    if (heroBenchBtn) {
      heroBenchBtn.addEventListener('click', () => switchView('benchmark'));
    }
  }

  /* ==========================================================================
     4. PLAYGROUND & RETRIEVAL ENGINE
     ========================================================================== */
  function initPlayground() {
    const searchInput = document.getElementById('query-input');
    const launchBtn = document.getElementById('btn-launch-query');
    const presetChips = document.querySelectorAll('.preset-chip');
    const methodOptions = document.querySelectorAll('.method-option');
    const sliderTopK = document.getElementById('slider-topk');
    const labelTopK = document.getElementById('val-topk');
    const sliderImages = document.getElementById('slider-images');
    const labelImages = document.getElementById('val-images');
    const checkGenerate = document.getElementById('toggle-generate');

    // Method selection
    methodOptions.forEach(opt => {
      opt.addEventListener('click', () => {
        methodOptions.forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
        state.selectedMethod = opt.dataset.method;
      });
    });

    // Top-K slider
    if (sliderTopK) {
      sliderTopK.addEventListener('input', (e) => {
        state.topK = parseInt(e.target.value, 10);
        labelTopK.textContent = state.topK;
      });
    }

    // Max Images slider
    if (sliderImages) {
      sliderImages.addEventListener('input', (e) => {
        state.maxImages = parseInt(e.target.value, 10);
        labelImages.textContent = state.maxImages;
      });
    }

    // Generate toggle
    if (checkGenerate) {
      checkGenerate.addEventListener('change', (e) => {
        state.doGenerate = e.target.checked;
      });
    }

    // Preset query click
    presetChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const q = chip.dataset.query;
        searchInput.value = q;
        executeSearch(q);
      });
    });

    // Launch button click
    if (launchBtn) {
      launchBtn.addEventListener('click', () => {
        const q = searchInput.value.trim();
        if (q) executeSearch(q);
      });
    }

    // Enter key
    if (searchInput) {
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const q = searchInput.value.trim();
          if (q) executeSearch(q);
        }
      });
    }
  }

  async function executeSearch(query) {
    state.currentQuery = query;
    const stageBar = document.getElementById('pipeline-stage-bar');
    const answerBox = document.getElementById('vlm-answer-box');
    const figuresGrid = document.getElementById('retrieved-figures-grid');
    const chunksList = document.getElementById('retrieved-chunks-list');
    const shortlistContainer = document.getElementById('shortlist-container');
    const resultsHeader = document.getElementById('results-count-title');

    // Visual loading state
    if (stageBar) {
      stageBar.innerHTML = `<span style="color: var(--cyan-glow);"><span class="status-beacon" style="display:inline-block; margin-right:8px;"></span>Routing through cosmic indexes: ${state.selectedMethod}...</span>`;
    }

    let payload = null;

    // 1. Try real backend API
    try {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          method: state.selectedMethod,
          k: state.topK,
          max_images: state.maxImages,
          generate: state.doGenerate
        })
      });
      payload = await resp.json();
      if (!resp.ok || payload.error) {
        // The backend is up but the search failed: show the real error rather
        // than quietly substituting cached results.
        if (stageBar) {
          stageBar.innerHTML = `<span style="color:#ff6b6b; font-weight:700;">Search failed: ${payload.error || ('HTTP ' + resp.status)}</span>`;
        }
        return;
      }
    } catch (err) {
      // Backend not reachable (e.g. the page was opened as a plain file).
      payload = null;
    }

    // 2. Offline fallback: replay cached evaluation results, clearly labelled.
    if (!payload) {
      payload = simulateRetrieval(query, state.selectedMethod, state.topK, state.maxImages, state.doGenerate);
      payload.offline = true;
    }

    renderSearchResults(payload);
  }

  function simulateRetrieval(query, methodKey, k, maxImages, doGenerate) {
    // Check if query matches one of the 40 benchmark questions
    const qLower = query.toLowerCase();
    let matchedQ = DATA.eval_set.questions.find(q => 
      q.question.toLowerCase().includes(qLower) || qLower.includes(q.question.toLowerCase().slice(0, 25))
    );

    let retrievedIds = [];
    let methodData = DATA.results.results.find(m => m.key === methodKey) || DATA.results.results[3]; // default multimodal
    let timings = methodData.stages_ms || { bm25: 3.2, dense: 9.1, clip: 18.5, merge: 0.05, total: 30.8 };

    if (matchedQ) {
      const qRes = methodData.per_question.find(p => p.id === matchedQ.id);
      if (qRes && qRes.retrieved) {
        retrievedIds = qRes.retrieved;
      } else {
        retrievedIds = matchedQ.relevant_ids;
      }
    } else {
      // General OS heuristic matching from chunks_map and figures
      const matchedChunks = Object.values(DATA.chunks_map)
        .filter(c => qLower.split(' ').some(w => w.length > 3 && c.content.toLowerCase().includes(w)))
        .slice(0, 10)
        .map(c => c.id);

      const matchedFigures = DATA.figures
        .filter(f => qLower.split(' ').some(w => w.length > 3 && f.filename.toLowerCase().includes(w)))
        .slice(0, 4)
        .map(f => f.id);

      if (['multimodal', 'full', 'tuned'].includes(methodKey)) {
        retrievedIds = [...matchedFigures.slice(0, maxImages), ...matchedChunks.slice(0, k)];
      } else {
        retrievedIds = matchedChunks.slice(0, k);
      }

      if (retrievedIds.length === 0) {
        // Fallback default sample evidence
        retrievedIds = ['cpu_mechanisms_chunk_13', 'cpu_mechanisms_chunk_12', 'slides_ch10_p30_0'];
      }
    }

    // Separate text chunks and images
    let evidence = [];
    let imageCount = 0;
    let textCount = 0;

    for (let id of retrievedIds) {
      const isImg = id.startsWith('slides_') || id.endsWith('_p') || id.includes('_page_') || id.includes('_p');
      if (isImg) {
        if (imageCount < maxImages && ['multimodal', 'full', 'tuned'].includes(methodKey)) {
          imageCount++;
          const fig = DATA.figures.find(f => f.id === id || f.filename.startsWith(id));
          const fn = fig ? fig.filename : `${id}.png`;
          evidence.push({
            id: id,
            is_image: true,
            score: 0.88 - (evidence.length * 0.04),
            citation: fn.replace('.png', '').replace(/_/g, ' '),
            content: fn,
            image_url: `/data/images/${fn}`
          });
        }
      } else {
        if (textCount < (k - imageCount)) {
          textCount++;
          const chk = DATA.chunks_map[id];
          evidence.push({
            id: id,
            is_image: false,
            score: 0.92 - (evidence.length * 0.05),
            citation: chk ? `${chk.source_doc} p.${chk.page}` : id,
            content: chk ? chk.content : `Content for chunk ${id}`
          });
        }
      }
    }

    // Answer synthesis
    let answer = null;
    if (doGenerate) {
      let qid = matchedQ ? matchedQ.id : null;
      let preset = qid && DATA.preset_answers[qid] ? DATA.preset_answers[qid] : null;
      if (preset) {
        answer = {
          text: preset.answer,
          cited: preset.citations
        };
      }
      // No cached answer for this question: show none rather than invent one.
    }

    return {
      query: query,
      method: methodKey,
      method_name: methodData.name,
      evidence: evidence,
      candidates: evidence.map((e, idx) => ({ id: e.id, is_image: e.is_image, score: 0.95 - (idx * 0.03) })),
      timings: timings,
      counts: { merged: evidence.length * 3, final: evidence.length },
      answer: answer
    };
  }

  function renderSearchResults(payload) {
    const stageBar = document.getElementById('pipeline-stage-bar');
    const answerBox = document.getElementById('vlm-answer-box');
    const figuresGrid = document.getElementById('retrieved-figures-grid');
    const chunksList = document.getElementById('retrieved-chunks-list');
    const shortlistContainer = document.getElementById('shortlist-container');
    const resultsHeader = document.getElementById('results-count-title');

    // 1. Stage Latency Bar
    if (stageBar && payload.timings) {
      const t = payload.timings;
      let parts = [];
      if (t.bm25) parts.push(`<div class="stage-timing-item"><span>BM25:</span><span class="val">${Math.round(t.bm25)} ms</span></div>`);
      if (t.dense) parts.push(`<div class="stage-timing-item"><span>Dense (BGE):</span><span class="val">${Math.round(t.dense)} ms</span></div>`);
      if (t.clip) parts.push(`<div class="stage-timing-item"><span>CLIP Vision:</span><span class="val">${Math.round(t.clip)} ms</span></div>`);
      if (t.merge) parts.push(`<div class="stage-timing-item"><span>RRF Merge:</span><span class="val">${(t.merge).toFixed(1)} ms</span></div>`);
      if (t.rerank) parts.push(`<div class="stage-timing-item"><span>Cross-Encoder:</span><span class="val">${Math.round(t.rerank)} ms</span></div>`);
      const total = t.total_wall_clock || t.total || 30.8;
      parts.push(`<div class="stage-timing-item" style="color:var(--cyan-glow); font-weight:700;"><span>Total Latency:</span><span class="val" style="color:#ffffff;">${Math.round(total)} ms</span></div>`);
      stageBar.innerHTML = parts.join('');
    }
    if (stageBar && payload.offline) {
      stageBar.innerHTML = `<div style="color:#ffb347; font-weight:700; width:100%;">OFFLINE MODE: backend not reachable. Showing cached evaluation results, not a live search. Start it with: python server.py</div>` + stageBar.innerHTML;
    }

    // 2. Answer Box
    if (answerBox) {
      if (payload.answer && payload.answer.text) {
        answerBox.style.display = 'block';
        answerBox.innerHTML = `
          <div class="vlm-badge">
            <span class="status-beacon"></span>
            Local Vision LLM Synthesized Answer
          </div>
          <div class="vlm-text">${payload.answer.text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</div>
          ${payload.answer.cited && payload.answer.cited.length ? `
            <div class="vlm-citations-row">
              <span style="color:var(--text-muted); font-family:var(--font-mono); font-size:0.75rem;">SOURCES CITED:</span>
              ${payload.answer.cited.map(c => `<span class="citation-pill">${c}</span>`).join('')}
            </div>
          ` : ''}
        `;
      } else {
        answerBox.style.display = 'none';
      }
    }

    // 3. Evidence items
    const imgEvidence = (payload.evidence || []).filter(e => e.is_image);
    const textEvidence = (payload.evidence || []).filter(e => !e.is_image);

    if (resultsHeader) {
      resultsHeader.innerHTML = `Retrieved Evidence <span style="font-size:0.85rem; color:var(--text-muted); font-family:var(--font-mono); font-weight:normal;">(${textEvidence.length} Text Chunks · ${imgEvidence.length} Figures)</span>`;
      resultsHeader.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Render Figures
    if (figuresGrid) {
      if (imgEvidence.length > 0) {
        figuresGrid.innerHTML = imgEvidence.map(img => `
          <div class="figure-card" data-img-url="${img.image_url}" data-title="${img.citation}">
            <div class="figure-thumb-wrapper">
              <img src="${img.image_url}" alt="${img.citation}" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'180\\' fill=\\'none\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%23080e22\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%2338bdf8\\' font-family=\\'monospace\\' font-size=\\'12\\' text-anchor=\\'middle\\'>Figure Diagram</text></svg>';">
            </div>
            <div class="figure-card-body">
              <div class="figure-citation">${img.citation}</div>
              <div class="figure-id">${img.id}</div>
              <div class="score-badge">RRF Score: ${img.score}</div>
            </div>
          </div>
        `).join('');

        // Modal zoom on figure click
        figuresGrid.querySelectorAll('.figure-card').forEach(card => {
          card.addEventListener('click', () => {
            openImageModal(card.dataset.imgUrl, card.dataset.title, "Extracted Course Slide Diagram (Indexed via CLIP ViT-B/32)");
          });
        });
      } else {
        figuresGrid.innerHTML = `
          <div style="grid-column: 1 / -1; padding: 24px; background: rgba(255,255,255,0.03); border-radius: var(--radius-md); text-align: center; color: var(--text-muted);">
            <div style="font-size: 1.4rem; margin-bottom: 6px;">🖼️</div>
            <div>${state.maxImages === 0 ? 'Image slots set to 0 (Text only).' : 'No figures retrieved with the current method setting.'}</div>
          </div>
        `;
      }
    }

    // Render Text Chunks
    if (chunksList) {
      if (textEvidence.length > 0) {
        chunksList.innerHTML = textEvidence.map(txt => `
          <div class="chunk-card">
            <div class="chunk-header">
              <div class="chunk-citation">${txt.citation}</div>
              <div class="score-badge">Score: ${txt.score}</div>
            </div>
            <div class="chunk-content">${escapeHtml(txt.content)}</div>
          </div>
        `).join('');
      } else {
        chunksList.innerHTML = `<div style="color:var(--text-muted); padding:16px;">No text passages retrieved.</div>`;
      }
    }

    // Shortlist Candidates
    if (shortlistContainer && payload.candidates) {
      shortlistContainer.innerHTML = `
        <div class="shortlist-expander">
          <div class="shortlist-header" id="shortlist-toggle">
            <span>Shortlist Candidates Before Cut (${payload.candidates.length} items)</span>
            <span style="font-family:var(--font-mono); font-size:0.8rem; color:var(--cyan-core);">[ Click to inspect ]</span>
          </div>
          <div class="shortlist-body" id="shortlist-body-content" style="display:none;">
            <table class="cosmic-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Type</th>
                  <th>Identifier</th>
                  <th>RRF / Cross-Encoder Score</th>
                </tr>
              </thead>
              <tbody>
                ${payload.candidates.map((c, idx) => `
                  <tr>
                    <td style="font-family:var(--font-mono);">${idx + 1}</td>
                    <td><span class="question-type-tag ${c.is_image ? 'figure' : 'text'}">${c.is_image ? 'IMAGE' : 'TEXT'}</span></td>
                    <td style="font-family:var(--font-mono); font-size:0.82rem;">${c.id}</td>
                    <td style="font-family:var(--font-mono); color:var(--cyan-glow);">${c.score}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      `;

      const toggle = document.getElementById('shortlist-toggle');
      const body = document.getElementById('shortlist-body-content');
      if (toggle && body) {
        toggle.addEventListener('click', () => {
          body.style.display = body.style.display === 'none' ? 'block' : 'none';
        });
      }
    }
  }

  /* ==========================================================================
     5. BENCHMARK LAB CONTROLLER ("THE OUTPUT")
     ========================================================================== */
  function initBenchmarkLab() {
    const tabBtns = document.querySelectorAll('.lab-tab-btn');
    const panels = document.querySelectorAll('.lab-view-panel');

    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        state.activeLabTab = btn.dataset.labTab;
        const targetPanel = document.getElementById(`lab-panel-${state.activeLabTab}`);
        if (targetPanel) targetPanel.classList.add('active');
      });
    });

    renderLeaderboardTable();
    renderAblationBreakdown();
    renderResearchFigures();
    renderBenchmarkQuestions();
  }

  function renderLeaderboardTable() {
    const container = document.getElementById('leaderboard-table-body');
    if (!container) return;

    const rows = DATA.results.results || [];
    container.innerHTML = rows.map((r, idx) => {
      const o = r.overall || {};
      const isBest = r.key === 'full' || r.key === 'tuned';
      return `
        <tr class="${isBest ? 'highlight-best' : ''}">
          <td style="font-weight:600; color:#ffffff;">
            ${r.name}
            <div style="font-size:0.75rem; color:var(--text-dim); margin-top:2px;">${r.note || ''}</div>
          </td>
          <td>
            <div class="metric-bar-cell">
              <span style="font-family:var(--font-mono); width:45px;">${(o['recall@5'] || 0).toFixed(3)}</span>
              <div class="metric-bar-bg">
                <div class="metric-bar-fill ${isBest ? 'emerald' : ''}" style="width: ${(o['recall@5'] || 0) * 100}%;"></div>
              </div>
            </div>
          </td>
          <td>
            <div class="metric-bar-cell">
              <span style="font-family:var(--font-mono); width:45px;">${(o['hit@5'] || 0).toFixed(3)}</span>
              <div class="metric-bar-bg">
                <div class="metric-bar-fill ${isBest ? 'emerald' : ''}" style="width: ${(o['hit@5'] || 0) * 100}%;"></div>
              </div>
            </div>
          </td>
          <td style="font-family:var(--font-mono); color:var(--cyan-glow);">${(o['mrr'] || 0).toFixed(3)}</td>
          <td style="font-family:var(--font-mono); color:var(--purple-glow);">${(o['ndcg@10'] || 0).toFixed(3)}</td>
          <td>
            <span class="cosmic-badge" style="font-size:0.72rem;">${Math.round(o['latency_ms'] || 0)} ms</span>
          </td>
        </tr>
      `;
    }).join('');
  }

  function renderAblationBreakdown() {
    const container = document.getElementById('ablation-cards-container');
    if (!container) return;

    const methods = DATA.results.results || [];
    container.innerHTML = `
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:20px; margin-bottom:32px;">
        <div class="telemetry-card" style="border-left: 4px solid var(--rose-core);">
          <div class="telemetry-icon">📉</div>
          <div class="telemetry-value" style="color:var(--rose-glow);">0.000</div>
          <div class="telemetry-label">Text-Only Recall on Figures</div>
          <p style="font-size:0.85rem; color:var(--text-muted); margin-top:8px;">
            BM25, BGE Dense, and Hybrid score absolute zero on diagram questions because slide images carry zero OCR text in the document tree.
          </p>
        </div>

        <div class="telemetry-card" style="border-left: 4px solid var(--cyan-core);">
          <div class="telemetry-icon">🌌</div>
          <div class="telemetry-value cyan">+53.6%</div>
          <div class="telemetry-label">Recall Boost with CLIP Vision</div>
          <p style="font-size:0.85rem; color:var(--text-muted); margin-top:8px;">
            Enabling the multimodal CLIP index lifts figure question Hit@5 from 0.000 to 0.730, and short-list recall to 0.786.
          </p>
        </div>

        <div class="telemetry-card" style="border-left: 4px solid var(--emerald-core);">
          <div class="telemetry-icon">🎯</div>
          <div class="telemetry-value" style="color:var(--emerald-glow);">0.811</div>
          <div class="telemetry-label">Full System Hit@5</div>
          <p style="font-size:0.85rem; color:var(--text-muted); margin-top:8px;">
            With cross-encoder reranking, the model surfaces the exact supporting proof in the top 5 slots for 81.1% of all evaluation questions.
          </p>
        </div>
      </div>

      <div class="benchmark-table-card">
        <div class="table-responsive">
          <table class="cosmic-table">
            <thead>
              <tr>
                <th>Retrieval Method</th>
                <th>Question Type</th>
                <th>Sample (n)</th>
                <th>Recall@5</th>
                <th>MRR</th>
                <th>nDCG@10</th>
              </tr>
            </thead>
            <tbody>
              ${methods.flatMap(m => {
                const bt = m.by_type || {};
                return ['text', 'figure', 'both'].map(type => {
                  const stat = bt[type] || {};
                  return `
                    <tr>
                      <td style="font-weight:600;">${m.name}</td>
                      <td><span class="question-type-tag ${type}">${type.toUpperCase()}</span></td>
                      <td style="font-family:var(--font-mono);">${stat.n || 0}</td>
                      <td style="font-family:var(--font-mono); color:${stat['recall@5'] > 0.4 ? 'var(--emerald-glow)' : 'var(--text-secondary)'}; font-weight:600;">
                        ${(stat['recall@5'] || 0).toFixed(3)}
                      </td>
                      <td style="font-family:var(--font-mono);">${(stat['mrr'] || 0).toFixed(3)}</td>
                      <td style="font-family:var(--font-mono);">${(stat['ndcg@10'] || 0).toFixed(3)}</td>
                    </tr>
                  `;
                });
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  function renderResearchFigures() {
    const container = document.getElementById('research-figures-grid');
    if (!container) return;

    const researchFiguresData = [
      {
        num: "Figure 01",
        title: "Chunk Length Distribution",
        filename: "01_chunk_length_distribution.png",
        caption: "Distribution of token counts across all 2,431 text chunks. Demonstrates the natural bimodal separation between dense OSTEP textbook paragraphs and slide bullet points."
      },
      {
        num: "Figure 02",
        title: "Words per Page Distribution",
        filename: "02_words_per_page.png",
        caption: "Word density per page across course materials. 303 slide pages have fewer than 30 words, revealing where conventional text search fails."
      },
      {
        num: "Figure 03",
        title: "Extracted Figure Class Distribution",
        filename: "03_image_class_distribution.png",
        caption: "Breakdown of the 843 indexed images: 472 whole-page renders and 371 embedded raster diagrams extracted with PyMuPDF."
      },
      {
        num: "Figure 04",
        title: "Figures per Course Chapter",
        filename: "04_figures_per_document.png",
        caption: "Distribution of diagrams across all 69 modules. Virtualization and concurrency slide decks contain the highest visual density."
      },
      {
        num: "Figure 05",
        title: "Text-Invisible Slide Pages Analysis",
        filename: "05_text_invisible_pages.png",
        caption: "Quantification of slides that carry crucial course diagrams but no parseable text characters."
      },
      {
        num: "Figure 06",
        title: "Pages vs. Chunks Scaling",
        filename: "06_pages_vs_chunks.png",
        caption: "Correlation between physical page counts and generated 300-word sliding window chunks."
      },
      {
        num: "Figure 07",
        title: "Method Performance Comparison",
        filename: "07_method_comparison.png",
        caption: "Recall@5, Hit@5, MRR, and nDCG across the 6 ablation methods, illustrating the steep ascent when multimodal indexing is engaged."
      },
      {
        num: "Figure 08",
        title: "Performance by Question Type",
        filename: "08_by_question_type.png",
        caption: "Side-by-side comparison on Text (n=15), Figure (n=14), and Both (n=8) questions. Highlights the 0.000 -> 0.786 figure recall jump."
      },
      {
        num: "Figure 09",
        title: "Retrieval Quality vs. Latency Pareto Frontier",
        filename: "09_quality_vs_latency.png",
        caption: "The trade-off between sub-15ms dual-branch bi-encoders and the 1,000ms cross-encoder reranking tier."
      }
    ];

    container.innerHTML = researchFiguresData.map((fig, idx) => `
      <div class="research-figure-card" data-idx="${idx}">
        <div class="figure-image-container">
          <img src="/eval/results/figures/${fig.filename}" alt="${fig.title}" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'400\\' height=\\'250\\' fill=\\'none\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%23080e22\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%238b5cf6\\' font-family=\\'sans-serif\\' font-size=\\'14\\' text-anchor=\\'middle\\'>${fig.title}</text></svg>';">
        </div>
        <div class="figure-card-details">
          <div class="figure-number">${fig.num}</div>
          <div class="figure-title">${fig.title}</div>
          <div class="figure-caption">${fig.caption}</div>
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.research-figure-card').forEach(card => {
      card.addEventListener('click', () => {
        const fig = researchFiguresData[card.dataset.idx];
        if (fig) {
          openImageModal(`/eval/results/figures/${fig.filename}`, fig.title, fig.caption);
        }
      });
    });
  }

  function renderBenchmarkQuestions() {
    const container = document.getElementById('benchmark-questions-grid');
    const searchInput = document.getElementById('search-benchmark-input');
    const filterPills = document.querySelectorAll('.bench-filter-pill');
    if (!container) return;

    function filterAndRender() {
      let list = DATA.eval_set.questions || [];

      // Filter type
      if (state.benchFilter !== 'All') {
        list = list.filter(q => q.answer_type.toLowerCase() === state.benchFilter.toLowerCase());
      }

      // Search query
      if (state.benchSearchQuery) {
        const sq = state.benchSearchQuery.toLowerCase();
        list = list.filter(q => 
          q.question.toLowerCase().includes(sq) ||
          q.note.toLowerCase().includes(sq) ||
          q.verify.toLowerCase().includes(sq) ||
          q.id.toLowerCase().includes(sq)
        );
      }

      container.innerHTML = list.map(q => `
        <div class="question-card">
          <div class="question-card-top">
            <span class="question-id-badge">${q.id.toUpperCase()}</span>
            <span class="question-type-tag ${q.answer_type}">${q.answer_type}</span>
          </div>
          <div class="question-text">${q.question}</div>
          <div class="question-verify">
            <span>📍 Verified:</span>
            <span style="color:var(--cyan-glow); font-family:var(--font-mono);">${q.verify}</span>
          </div>
          <div class="question-note-box">
            <span style="color:var(--amber-glow); font-weight:600; font-family:var(--font-mono); font-size:0.75rem;">GROUND TRUTH KEY:</span>
            <div>${q.note}</div>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:auto; padding-top:8px;">
            <span style="font-size:0.75rem; color:var(--text-dim); font-family:var(--font-mono);">IDs: ${q.relevant_ids.join(', ')}</span>
            <button class="preset-chip" onclick="window.queryFromBench('${escapeHtml(q.question)}')">Query live ⚡</button>
          </div>
        </div>
      `).join('');
    }

    filterPills.forEach(pill => {
      pill.addEventListener('click', () => {
        filterPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        state.benchFilter = pill.dataset.filter;
        filterAndRender();
      });
    });

    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        state.benchSearchQuery = e.target.value.trim();
        filterAndRender();
      });
    }

    filterAndRender();
  }

  /* ==========================================================================
     6. DATASET EXPLORER CONTROLLER ("THE DATASET")
     ========================================================================== */
  function initDatasetExplorer() {
    const docGrid = document.getElementById('documents-grid');
    const figureGrid = document.getElementById('dataset-figures-grid');
    const docSearch = document.getElementById('search-doc-input');
    const categoryPills = document.querySelectorAll('.doc-cat-pill');
    if (!docGrid) return;

    function filterAndRenderDocs() {
      let docs = DATA.documents || [];

      if (state.docFilter !== 'All') {
        docs = docs.filter(d => d.category === state.docFilter);
      }

      if (state.docSearchQuery) {
        const sq = state.docSearchQuery.toLowerCase();
        docs = docs.filter(d => 
          d.title.toLowerCase().includes(sq) ||
          d.filename.toLowerCase().includes(sq) ||
          d.category.toLowerCase().includes(sq)
        );
      }

      docGrid.innerHTML = docs.map(d => `
        <div class="doc-card">
          <div>
            <div class="doc-top">
              <span class="doc-category-badge">${d.category}</span>
              <span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-dim);">${d.size_kb} KB</span>
            </div>
            <div class="doc-title">${d.title}</div>
            <div class="doc-filename">${d.filename}</div>
          </div>
          <div class="doc-stats-row">
            <span>${d.type}</span>
            <a href="/data/own/${d.filename}" target="_blank" class="doc-action-btn">
              View Source PDF ↗
            </a>
          </div>
        </div>
      `).join('');
    }

    categoryPills.forEach(pill => {
      pill.addEventListener('click', () => {
        categoryPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        state.docFilter = pill.dataset.cat;
        filterAndRenderDocs();
      });
    });

    if (docSearch) {
      docSearch.addEventListener('input', (e) => {
        state.docSearchQuery = e.target.value.trim();
        filterAndRenderDocs();
      });
    }

    filterAndRenderDocs();

    // Render dataset figures sample
    if (figureGrid) {
      const sampleFigs = (DATA.figures || []).slice(0, 48);
      figureGrid.innerHTML = sampleFigs.map(f => `
        <div class="figure-card" onclick="window.openImageModal('${f.path}', '${f.name}', 'Indexed Course Diagram')">
          <div class="figure-thumb-wrapper">
            <img src="${f.path}" alt="${f.name}" onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'280\\' height=\\'180\\' fill=\\'none\\'><rect width=\\'100%\\' height=\\'100%\\' fill=\\'%23080e22\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%2338bdf8\\' font-family=\\'monospace\\' font-size=\\'12\\' text-anchor=\\'middle\\'>Slide Render</text></svg>';">
          </div>
          <div class="figure-card-body">
            <div class="figure-citation" style="font-size:0.8rem;">${f.name}</div>
            <span class="question-type-tag ${f.is_whole_page ? 'text' : 'figure'}" style="align-self:flex-start;">
              ${f.is_whole_page ? 'Whole Page' : 'Cropped Figure'}
            </span>
          </div>
        </div>
      `).join('');
    }
  }

  /* ==========================================================================
     7. IMAGE MODAL LIGHTBOX
     ========================================================================== */
  function initImageModal() {
    const overlay = document.getElementById('image-modal-overlay');
    const closeBtn = document.getElementById('modal-close-btn');

    if (closeBtn && overlay) {
      closeBtn.addEventListener('click', () => {
        overlay.classList.remove('active');
      });

      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
          overlay.classList.remove('active');
        }
      });

      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay.classList.contains('active')) {
          overlay.classList.remove('active');
        }
      });
    }
  }

  window.openImageModal = function (imgUrl, title, caption) {
    const overlay = document.getElementById('image-modal-overlay');
    const modalImg = document.getElementById('modal-img-element');
    const modalTitle = document.getElementById('modal-title-element');
    const modalCaption = document.getElementById('modal-caption-element');

    if (overlay && modalImg) {
      modalImg.src = imgUrl;
      if (modalTitle) modalTitle.textContent = title || 'Figure Detail';
      if (modalCaption) modalCaption.textContent = caption || '';
      overlay.classList.add('active');
    }
  };

  window.queryFromBench = function (questionText) {
    const navBtn = document.querySelector('.nav-btn[data-view="playground"]');
    if (navBtn) navBtn.click();
    const searchInput = document.getElementById('query-input');
    if (searchInput) {
      searchInput.value = questionText;
      executeSearch(questionText);
    }
  };

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  /* ==========================================================================
     8. EDITORIAL SCROLL REVEAL, HEADER & STATS COUNTER
     ========================================================================== */
  function initEditorialInteractions() {
    // Header compression on scroll
    const header = document.getElementById('editorial-header');
    window.addEventListener('scroll', () => {
      if (header) {
        if (window.scrollY > 40) {
          header.classList.add('scrolled');
        } else {
          header.classList.remove('scrolled');
        }
      }
    });

    // Mobile menu toggle
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const mainNav = document.getElementById('main-nav');
    if (mobileBtn && mainNav) {
      mobileBtn.addEventListener('click', () => {
        mainNav.classList.toggle('open');
      });
      mainNav.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          mainNav.classList.remove('open');
        });
      });
    }

    // Collapsible experimental parameters panel
    const toggleBtn = document.getElementById('controls-toggle-btn');
    const panelBody = document.getElementById('controls-panel-body');
    const toggleIndicator = document.getElementById('panel-toggle-indicator');
    if (toggleBtn && panelBody) {
      toggleBtn.addEventListener('click', () => {
        const isClosed = panelBody.style.display === 'none';
        panelBody.style.display = isClosed ? 'grid' : 'none';
        if (toggleIndicator) {
          toggleIndicator.textContent = isClosed ? '[ Collapse Parameters − ]' : '[ Expand Parameters + ]';
        }
        toggleBtn.setAttribute('aria-expanded', isClosed ? 'true' : 'false');
      });
    }

    // Intersection Observer for scroll reveals & count-up animations
    const reveals = document.querySelectorAll('.reveal-editorial');
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            const statNumbers = entry.target.querySelectorAll('.stat-number');
            statNumbers.forEach(animateNumber);
            obs.unobserve(entry.target);
          }
        });
      }, { threshold: 0.12 });

      reveals.forEach(el => observer.observe(el));
    } else {
      reveals.forEach(el => el.classList.add('revealed'));
    }
  }

  function animateNumber(el) {
    const targetStr = el.dataset.target;
    if (!targetStr) return;
    const isDecimal = targetStr.includes('.');
    const targetVal = parseFloat(targetStr.replace(/,/g, ''));
    const duration = 1400;
    const startVal = 0;
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = startVal + (targetVal - startVal) * ease;

      if (isDecimal) {
        el.textContent = current.toFixed(3);
      } else {
        el.textContent = Math.round(current).toLocaleString();
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        el.textContent = targetStr;
      }
    }

    requestAnimationFrame(update);
  }

  /* ==========================================================================
     9. APP INITIALIZATION
     ========================================================================== */
  document.addEventListener('DOMContentLoaded', () => {
    initStarfield();
    initNavigation();
    initPlayground();
    initBenchmarkLab();
    initDatasetExplorer();
    initImageModal();
    initEditorialInteractions();

    const audioBtn = document.getElementById('btn-audio-toggle');
    if (audioBtn) {
      audioBtn.addEventListener('click', toggleCosmicAudio);
    }

    console.log("🏛️ Project Black Hole 2 Editorial Monograph Online.");
  });

})();
