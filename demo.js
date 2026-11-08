'use strict';
(() => {
  const cases = JSON.parse(document.getElementById('demo-data').textContent);
  const $ = id => document.getElementById(id);
  const audio = $('listening-audio');
  const labels = {original: 'Original model', main: 'Edited model', prompt: 'Ask for flute', experimental: 'Experimental edit'};
  const descriptions = {original: 'Original model output', main: 'Edited model output', prompt: 'Original model · new prompt', experimental: 'Experimental model output'};
  let selectedCase = cases[0];
  let selectedVariant = 'original';
  let selectionVersion = 0;
  const formatTime = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  const clip = () => selectedCase.clips[selectedVariant];

  function updatePlayback() {
    const duration = Number.isFinite(audio.duration) ? audio.duration : clip().duration;
    const current = Math.min(audio.currentTime || 0, duration);
    $('playback-progress').max = duration;
    $('playback-progress').value = current;
    $('time-label').textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    $('play-label').textContent = audio.paused ? 'Play' : 'Pause';
    $('play-button').setAttribute('aria-label', `${audio.paused ? 'Play' : 'Pause'} selected recording`);
    $('play-button').querySelector('.play-icon').textContent = audio.paused ? '▶' : 'Ⅱ';
    $('waveform').querySelectorAll('rect').forEach((bar, index) => {
      bar.classList.toggle('played', index / 100 < current / duration);
    });
  }

  async function play() {
    const version = selectionVersion;
    $('playback-error').hidden = true;
    try { await audio.play(); }
    catch (error) {
      if (version !== selectionVersion || error.name === 'AbortError') return;
      $('playback-error').textContent = 'Could not play this recording. Try Play again. If you opened the ZIP locally, unzip the whole folder first.';
      $('playback-error').hidden = false;
    }
    updatePlayback();
  }

  function setVariant(variant, continuePlaying = false) {
    selectionVersion += 1;
    audio.pause();
    selectedVariant = variant;
    const current = clip();
    $('variant-title').textContent = descriptions[variant];
    $('duration-label').textContent = formatTime(current.duration);
    $('heard').textContent = selectedCase.heard[variant];
    const isText = typeof current.prompt === 'string';
    $('input-label').textContent = isText ? 'Prompt sent to the model' : 'Recording given to the model';
    $('input-text').textContent = isText ? current.prompt : selectedCase.request;
    $('comparison-note').textContent = variant === 'prompt'
      ? 'Here, the original model gets a different prompt that asks for flute.'
      : isText
        ? 'Original and edited models get this exact prompt. Only the saved weights change.'
        : 'Every version gets the same recording as a style reference and generates new music.';
    $('seed-label').textContent = `Generation seed: ${current.seed}. All versions use the same seed and generation settings.`;
    $('variant-buttons').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.variant === variant)));
    const ns = 'http://www.w3.org/2000/svg';
    const bars = current.peaks.map((peak, index) => {
      const rect = document.createElementNS(ns, 'rect');
      const height = Math.max(2, peak * 110);
      Object.entries({x: index * 6, y: (120 - height) / 2, width: 3, height, rx: 1, class: 'wave-bar'}).forEach(([key, value]) => rect.setAttribute(key, value));
      return rect;
    });
    $('waveform').replaceChildren(...bars);
    $('playback-error').hidden = true;
    audio.src = document.documentElement.dataset.assetBase + current.src;
    audio.load();
    updatePlayback();
    if (continuePlaying) play();
  }

  function setCase(id) {
    selectedCase = cases.find(item => item.id === id);
    $('case-select').value = id;
    $('case-kind').textContent = selectedCase.kind;
    $('case-summary').textContent = selectedCase.summary;
    const buttons = Object.keys(labels).filter(variant => selectedCase.clips[variant]).map(variant => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.variant = variant;
      button.textContent = labels[variant];
      button.addEventListener('click', () => setVariant(variant, !audio.paused));
      return button;
    });
    $('variant-buttons').replaceChildren(...buttons);
    setVariant('original');
  }

  $('case-select').disabled = false;
  $('case-select').addEventListener('change', event => setCase(event.target.value));
  $('play-button').hidden = false;
  $('play-button').addEventListener('click', () => audio.paused ? play() : audio.pause());
  ['play', 'pause', 'timeupdate', 'loadedmetadata', 'ended'].forEach(event => audio.addEventListener(event, updatePlayback));
  audio.addEventListener('error', () => {
    $('playback-error').textContent = 'Recording unavailable. Unzip the entire bundle and keep the audio folder beside this page.';
    $('playback-error').hidden = false;
    updatePlayback();
  });
  document.querySelectorAll('audio').forEach(player => player.addEventListener('play', () => {
    document.querySelectorAll('audio').forEach(other => { if (other !== player) other.pause(); });
  }));
  const weights = JSON.parse($('weight-data').textContent);
  const cells = [...$('matrix-explorer').querySelectorAll('rect')];
  let selectedBlock = 0;
  const color = (value, limit) => {
    const amount = Math.sqrt(Math.min(1, Math.abs(value) / limit));
    const ink = value < 0 ? [181, 45, 87] : [36, 84, 214];
    return `rgb(${ink.map(channel => Math.round(247 + (channel - 247) * amount)).join(',')})`;
  };
  const number = value => `${value >= 0 ? '+' : ''}${value.toFixed(6)}`;
  function inspectBlock(index) {
    cells[selectedBlock].classList.remove('inspected');
    selectedBlock = index;
    cells[index].classList.add('inspected');
    const row = Math.floor(index / 64) * 16;
    const column = index % 64 * 16;
    $('block-position').textContent = `Rows ${row}–${row + 15} · Columns ${column}–${column + 15}`;
    ['before', 'after', 'delta'].forEach(mode => { $('weight-' + mode).textContent = number(weights[mode][index]); });
    $('weight-delta').style.color = weights.delta[index] < 0 ? 'var(--rose)' : 'var(--blue)';
    const barLimit = weights.weight_scale;
    ['before', 'after'].forEach(mode => {
      const value = weights[mode][index];
      const width = Math.abs(value) / barLimit * 50;
      const bar = $(mode + '-bar');
      bar.style.width = width + '%';
      bar.style.left = (value < 0 ? 50 - width : 50) + '%';
      bar.style.background = value < 0 ? 'var(--rose)' : 'var(--blue)';
    });
    $('weight-readout').textContent = 'Mean of the 256 weights in this block.';
  }
  function setWeightMode(mode) {
    const scale = mode === 'delta' ? weights.scale_max_abs : weights.weight_scale;
    cells.forEach((cell, index) => cell.setAttribute('fill', color(weights[mode][index], scale)));
    $('matrix-modes').querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
    const viewLabel = {before:'Original weights',after:'Edited weights',delta:'Weight change'}[mode];
    $('matrix-view-label').textContent = `${viewLabel} · contrast enhanced`;
    $('matrix-explorer').querySelector('svg').setAttribute('aria-label', `${viewLabel}, averaged over 16 by 16 weight blocks`);
  }
  $('matrix-modes').querySelectorAll('button').forEach(button => button.addEventListener('click', () => setWeightMode(button.dataset.mode)));
  ['pointerover', 'click'].forEach(eventName => {
    $('matrix-explorer').addEventListener(eventName, event => {
      const index = event.target.dataset.index;
      if (index !== undefined) inspectBlock(Number(index));
    });
  });
  $('matrix-explorer').addEventListener('keydown', event => {
    const steps = {ArrowLeft:-1,ArrowRight:1,ArrowUp:-64,ArrowDown:64};
    if (steps[event.key] !== undefined) {
      event.preventDefault();
      inspectBlock(Math.max(0, Math.min(cells.length - 1, selectedBlock + steps[event.key])));
    }
  });
  setWeightMode('delta');
  inspectBlock(0);
  const matrixStage = $('matrix-stage');
  const overview = $('model-overview');
  const modelMap = overview.querySelector('.model-map');
  const targetTile = overview.querySelector('.edited-tensor');
  const targetBox = modelMap.dataset.targetViewbox.split(' ').map(Number);
  const replayZoom = $('replay-zoom');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let zoomFrame = 0;
  function finishZoom() {
    cancelAnimationFrame(zoomFrame);
    zoomFrame = 0;
    overview.hidden = true;
    matrixStage.classList.remove('zooming');
    replayZoom.disabled = false;
  }
  function playZoom() {
    if (reducedMotion.matches || replayZoom.disabled) return;
    modelMap.setAttribute('viewBox', '0 0 512 384');
    overview.style.setProperty('--label-opacity', '1');
    targetTile.style.opacity = '1';
    overview.hidden = false;
    replayZoom.disabled = true;
    matrixStage.classList.add('zooming');
    const started = performance.now();
    const [x, y, width, height] = targetBox;
    function frame(now) {
      const progress = Math.max(0, Math.min(1, (now - started - 1600) / 2600));
      const eased = progress * progress * (3 - 2 * progress);
      const scale = Math.pow(width / 512, eased);
      // Move the tile toward screen center while its own weight pattern expands.
      const centerX = x + width / 2 + (256 - x - width / 2) * scale * (1 - eased);
      const centerY = y + height / 2 + (192 - y - height / 2) * scale * (1 - eased);
      modelMap.setAttribute('viewBox', `${centerX - 256 * scale} ${centerY - 192 * scale} ${512 * scale} ${384 * scale}`);
      overview.style.setProperty('--label-opacity', String(Math.max(0, 1 - progress / 0.22)));
      targetTile.style.opacity = String(Math.max(0, 1 - progress / 0.65));
      if (progress < 1) zoomFrame = requestAnimationFrame(frame);
      else finishZoom();
    }
    zoomFrame = requestAnimationFrame(frame);
  }
  reducedMotion.addEventListener('change', finishZoom);
  replayZoom.hidden = false;
  replayZoom.addEventListener('click', playZoom);
  const zoomObserver = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) {
      zoomObserver.disconnect();
      playZoom();
    }
  }, {threshold: 0.45});
  zoomObserver.observe(matrixStage);
  document.documentElement.classList.add('enhanced');
  setCase(cases[0].id);
})();

(() => {
  const documents = JSON.parse(document.getElementById('document-data').textContent);
  const reader = document.getElementById('document-reader');
  const content = document.getElementById('document-content');
  const select = document.getElementById('document-select');
  const originalTitle = document.title;
  let returnFocus = null;
  let renderedHash = null;

  function route(hash) {
    const params = new URLSearchParams(hash.slice(1));
    return params.has('doc') ? {name: params.get('doc'), section: params.get('section')} : null;
  }

  function renderDocument() {
    const current = route(location.hash);
    if (!current) {
      if (reader.open) {
        reader.close();
        document.body.classList.remove('document-open');
        returnFocus?.focus({preventScroll: true});
      }
      document.title = originalTitle;
      renderedHash = null;
      return;
    }
    if (renderedHash === location.hash && reader.open) return;
    renderedHash = location.hash;
    const record = Object.hasOwn(documents, current.name) ? documents[current.name] : null;
    if (record) {
      content.innerHTML = record.html;
      select.value = current.name;
      document.title = `${record.title} — Piano to flute`;
    } else {
      const message = document.createElement('p');
      message.textContent = 'This document is not in the bundle. Choose another document above.';
      content.replaceChildren(message);
      select.selectedIndex = -1;
      document.title = `Document unavailable — Piano to flute`;
    }
    if (!reader.open) {
      returnFocus = document.activeElement;
      reader.showModal();
      document.body.classList.add('document-open');
    }
    content.scrollTop = 0;
    const target = current.section && document.getElementById(`doc-${current.section}`);
    const section = target && content.contains(target) ? target : null;
    if (section) section.scrollIntoView({block: 'start', behavior: 'instant'});
    const focusTarget = section || content.querySelector('h1') || content;
    focusTarget.setAttribute('tabindex', '-1');
    focusTarget.focus({preventScroll: true});
  }

  function navigateDocument(hash) {
    if (hash !== location.hash) {
      const alreadyReading = route(location.hash);
      const depth = history.state?.docDepth || 0;
      history.pushState({
        // Direct document links have no earlier demo entry to return to.
        docDepth: alreadyReading ? (depth > 0 ? depth + 1 : 0) : 1
      }, '', hash);
    } else {
      renderedHash = null;
    }
    renderDocument();
  }

  function closeDocument() {
    if (history.state?.docDepth > 0) history.go(-history.state.docDepth);
    else {
      history.replaceState(null, '', '#top');
      renderDocument();
    }
  }

  for (const [name, record] of Object.entries(documents)) {
    select.add(new Option(record.title, name));
  }
  select.addEventListener('change', () => navigateDocument('#' + new URLSearchParams({doc: select.value})));
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href]');
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const hash = link.getAttribute('href');
    if (!hash.startsWith('#') || !route(hash)) return;
    event.preventDefault();
    navigateDocument(hash);
  });
  document.getElementById('document-close').addEventListener('click', closeDocument);
  reader.addEventListener('cancel', event => { event.preventDefault(); closeDocument(); });
  reader.addEventListener('keydown', event => {
    if (event.key !== 'Tab') return;
    const controls = [...reader.querySelectorAll('a[href], button:not([disabled]), select, [tabindex="0"]')];
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && (document.activeElement === first || !controls.includes(document.activeElement))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
  let pressedBackdrop = false;
  reader.addEventListener('pointerdown', event => { pressedBackdrop = event.target === reader; });
  reader.addEventListener('click', event => {
    const bounds = reader.getBoundingClientRect();
    if (pressedBackdrop && event.target === reader &&
        (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) closeDocument();
    pressedBackdrop = false;
  });
  window.addEventListener('popstate', renderDocument);
  window.addEventListener('hashchange', renderDocument);
  renderDocument();
})();
