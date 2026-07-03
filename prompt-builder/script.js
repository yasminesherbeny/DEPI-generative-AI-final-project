// Prompt Builder — interactivity for the DEPI Generative AI Final Project
// Builds structured prompts for the GPT-2 LoRA product description model.

(() => {
  'use strict';

  const HISTORY_KEY = 'promptBuilder.promptHistory';
  const THEME_KEY = 'promptBuilder.theme';

  const TEMPLATE_PRESETS = {
    ecommerce: { tone: 'Professional', length: 'Medium (100 words)', style: 'Formal', cta: true },
    social: { tone: 'Playful', length: 'Short (50 words)', style: 'Conversational', cta: true },
    technical: { tone: 'Technical', length: 'Long (200 words)', style: 'Data-Driven', cta: false },
    story: { tone: 'Storytelling', length: 'Long (200 words)', style: 'Creative', cta: false },
    luxury: { tone: 'Luxurious', length: 'Medium (100 words)', style: 'Formal', cta: true },
    budget: { tone: 'Persuasive', length: 'Short (50 words)', style: 'Conversational', cta: true },
  };

  const state = {
    tone: 'Professional',
    length: 'Medium (100 words)',
    style: 'Formal',
    cta: true,
    activeTemplate: null,
  };

  const params = {
    temperature: 0.7,
    topK: 50,
    topP: 0.95,
    maxLength: 150,
    numVariations: 1,
  };

  const $ = (id) => document.getElementById(id);

  const els = {
    productName: $('productName'),
    brandName: $('brandName'),
    productCategory: $('productCategory'),
    targetAudience: $('targetAudience'),
    keyFeatures: $('keyFeatures'),
    priceRange: $('priceRange'),
    promptCode: $('promptCode'),
    charCount: $('charCount'),
    pythonPanel: $('pythonPanel'),
    pythonCode: $('pythonCode'),
    historySidebar: $('historySidebar'),
    sidebarOverlay: $('sidebarOverlay'),
    historyList: $('historyList'),
    historyEmpty: $('historyEmpty'),
    historyBadge: $('historyBadge'),
    toastContainer: $('toastContainer'),
  };

  // ---------- Prompt assembly ----------

  function buildPrompt() {
    const lines = ['Generate a product description with the following specifications:', ''];

    const name = els.productName.value.trim();
    const brand = els.brandName.value.trim();
    const category = els.productCategory.value;
    const audience = els.targetAudience.value.trim();
    const features = els.keyFeatures.value.trim();
    const price = els.priceRange.value.trim();

    lines.push(`Product Name: ${name || '—'}`);
    if (brand) lines.push(`Brand: ${brand}`);
    if (category) lines.push(`Category: ${category}`);
    if (audience) lines.push(`Target Audience: ${audience}`);
    if (features) lines.push(`Key Features: ${features}`);
    if (price) lines.push(`Price Range: ${price}`);

    lines.push('');
    lines.push(`Tone: ${state.tone}`);
    lines.push(`Length: ${state.length}`);
    lines.push(`Style: ${state.style}`);
    lines.push(`Include Call-to-Action: ${state.cta ? 'Yes' : 'No'}`);
    lines.push('');
    lines.push('Description:');

    return lines.join('\n');
  }

  function updatePreview() {
    const prompt = buildPrompt();
    els.promptCode.textContent = prompt;
    els.charCount.textContent = `${prompt.length} characters`;
    return prompt;
  }

  function buildPythonCode(prompt) {
    return `from src.inference.generate import load_generator, generate_text

generator = load_generator("lora_model")

prompt = """${prompt}"""

output = generator(
    prompt,
    max_length=${params.maxLength},
    temperature=${params.temperature},
    top_k=${params.topK},
    top_p=${params.topP},
    num_return_sequences=${params.numVariations},
    do_sample=True,
)

for i, result in enumerate(output):
    print(f"--- Variation {i + 1} ---")
    print(result["generated_text"])
`;
  }

  // ---------- Toasts ----------

  function showToast(message, icon = '✅') {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span class="toast__icon">${icon}</span><span class="toast__text">${message}</span>`;
    els.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('toast--exit');
      toast.addEventListener('animationend', () => toast.remove(), { once: true });
    }, 2800);
  }

  // ---------- Chip groups (tone / length / style) ----------

  function setupChipGroup(groupId, stateKey) {
    const group = $(groupId);
    if (!group) return;
    group.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip');
      if (!chip || !group.contains(chip)) return;
      group.querySelectorAll('.chip').forEach((c) => {
        c.classList.remove('chip--active');
        c.setAttribute('aria-checked', 'false');
      });
      chip.classList.add('chip--active');
      chip.setAttribute('aria-checked', 'true');
      state[stateKey] = chip.dataset.value;
      state.activeTemplate = null;
      clearTemplateSelection();
      updatePreview();
    });
  }

  function clearTemplateSelection() {
    document.querySelectorAll('.template-card').forEach((c) => c.classList.remove('template-card--active'));
  }

  // ---------- Templates ----------

  function applyTemplate(name) {
    const preset = TEMPLATE_PRESETS[name];
    if (!preset) return;

    setChipActive('toneGroup', preset.tone);
    setChipActive('lengthGroup', preset.length);
    setChipActive('styleGroup', preset.style);
    setCta(preset.cta);

    state.tone = preset.tone;
    state.length = preset.length;
    state.style = preset.style;
    state.cta = preset.cta;
    state.activeTemplate = name;

    clearTemplateSelection();
    const card = document.querySelector(`[data-template="${name}"]`);
    if (card) card.classList.add('template-card--active');

    updatePreview();
    showToast(`Applied "${card ? card.querySelector('.template-card__label').textContent : name}" template`);
  }

  function setChipActive(groupId, value) {
    const group = $(groupId);
    if (!group) return;
    group.querySelectorAll('.chip').forEach((c) => {
      const active = c.dataset.value === value;
      c.classList.toggle('chip--active', active);
      c.setAttribute('aria-checked', String(active));
    });
  }

  function setCta(enabled) {
    const toggle = $('ctaToggle');
    const label = $('ctaLabel');
    toggle.classList.toggle('toggle--active', enabled);
    toggle.setAttribute('aria-checked', String(enabled));
    label.textContent = enabled ? 'Enabled' : 'Disabled';
    state.cta = enabled;
  }

  function setupTemplates() {
    document.querySelectorAll('.template-card').forEach((card) => {
      card.addEventListener('click', () => applyTemplate(card.dataset.template));
    });
  }

  // ---------- CTA toggle ----------

  function setupCtaToggle() {
    $('ctaToggle').addEventListener('click', () => {
      setCta(!state.cta);
      state.activeTemplate = null;
      clearTemplateSelection();
      updatePreview();
    });
  }

  // ---------- Advanced parameters ----------

  function setupAdvancedPanel() {
    const toggle = $('advancedToggle');
    const body = $('advancedBody');
    toggle.addEventListener('click', () => {
      const expanded = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', String(!expanded));
      body.setAttribute('aria-hidden', String(expanded));
    });

    const sliderConfig = [
      ['temperature', 'temperatureValue', parseFloat],
      ['topK', 'topKValue', parseInt],
      ['topP', 'topPValue', parseFloat],
      ['maxLength', 'maxLengthValue', parseInt],
      ['numVariations', 'numVariationsValue', parseInt],
    ];

    sliderConfig.forEach(([inputId, labelId, parse]) => {
      const input = $(inputId);
      const label = $(labelId);
      input.addEventListener('input', () => {
        params[inputId] = parse(input.value);
        label.textContent = input.value;
      });
    });
  }

  // ---------- Form fields ----------

  function setupFormFields() {
    ['productName', 'brandName', 'productCategory', 'targetAudience', 'keyFeatures', 'priceRange'].forEach((id) => {
      els[id].addEventListener('input', updatePreview);
      els[id].addEventListener('change', updatePreview);
    });
  }

  // ---------- Actions ----------

  async function copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
      return true;
    }
  }

  function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function currentSnapshot() {
    return {
      prompt: updatePreview(),
      fields: {
        productName: els.productName.value.trim(),
        brandName: els.brandName.value.trim(),
        productCategory: els.productCategory.value,
        targetAudience: els.targetAudience.value.trim(),
        keyFeatures: els.keyFeatures.value.trim(),
        priceRange: els.priceRange.value.trim(),
      },
      style: { tone: state.tone, length: state.length, style: state.style, cta: state.cta },
      generation: { ...params },
      timestamp: new Date().toISOString(),
    };
  }

  function setupActions() {
    $('btnCopyPrompt').addEventListener('click', async () => {
      const prompt = updatePreview();
      await copyToClipboard(prompt);
      showToast('Prompt copied to clipboard');
    });

    $('btnCopyPython').addEventListener('click', async () => {
      const prompt = updatePreview();
      const code = buildPythonCode(prompt);
      els.pythonCode.textContent = code;
      els.pythonPanel.classList.remove('hidden');
      await copyToClipboard(code);
      showToast('Python code copied to clipboard');
    });

    $('closePython').addEventListener('click', () => {
      els.pythonPanel.classList.add('hidden');
    });

    $('btnDownloadJSON').addEventListener('click', () => {
      const snapshot = currentSnapshot();
      downloadJSON(snapshot, `prompt-${Date.now()}.json`);
      showToast('Prompt downloaded as JSON');
    });

    $('btnSaveHistory').addEventListener('click', () => {
      saveToHistory();
      showToast('Saved to history');
    });

    $('btnResetAll').addEventListener('click', () => {
      resetAll();
      showToast('All fields reset', '♻️');
    });
  }

  // ---------- History ----------

  function loadHistory() {
    try {
      return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch {
      return [];
    }
  }

  function persistHistory(history) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }

  function saveToHistory() {
    const snapshot = currentSnapshot();
    const history = loadHistory();
    history.unshift({
      id: `${Date.now()}`,
      name: snapshot.fields.productName || 'Untitled prompt',
      ...snapshot,
    });
    persistHistory(history);
    renderHistory();
  }

  function deleteHistoryItem(id) {
    const history = loadHistory().filter((item) => item.id !== id);
    persistHistory(history);
    renderHistory();
  }

  function loadHistoryItem(id) {
    const item = loadHistory().find((h) => h.id === id);
    if (!item) return;

    els.productName.value = item.fields.productName || '';
    els.brandName.value = item.fields.brandName || '';
    els.productCategory.value = item.fields.productCategory || '';
    els.targetAudience.value = item.fields.targetAudience || '';
    els.keyFeatures.value = item.fields.keyFeatures || '';
    els.priceRange.value = item.fields.priceRange || '';

    setChipActive('toneGroup', item.style.tone);
    setChipActive('lengthGroup', item.style.length);
    setChipActive('styleGroup', item.style.style);
    setCta(item.style.cta);
    state.tone = item.style.tone;
    state.length = item.style.length;
    state.style = item.style.style;
    clearTemplateSelection();

    Object.entries(item.generation || {}).forEach(([key, value]) => {
      const input = $(key);
      const label = $(`${key}Value`);
      if (input) {
        input.value = value;
        params[key] = value;
      }
      if (label) label.textContent = value;
    });

    updatePreview();
    closeHistorySidebar();
    showToast('Prompt loaded from history');
  }

  function renderHistory() {
    const history = loadHistory();
    els.historyList.querySelectorAll('.history-item').forEach((n) => n.remove());
    els.historyEmpty.classList.toggle('hidden', history.length > 0);
    els.historyBadge.classList.toggle('hidden', history.length === 0);
    els.historyBadge.textContent = String(history.length);

    history.forEach((item) => {
      const node = document.createElement('div');
      node.className = 'history-item';
      node.setAttribute('role', 'listitem');
      const date = new Date(item.timestamp).toLocaleString();
      node.innerHTML = `
        <div class="history-item__header">
          <span class="history-item__name">${escapeHtml(item.name)}</span>
          <span class="history-item__date">${date}</span>
        </div>
        <div class="history-item__preview">${escapeHtml(item.prompt.slice(0, 140))}</div>
        <div class="history-item__actions">
          <button class="btn btn--outline btn--sm" data-action="load">Load</button>
          <button class="btn btn--ghost btn--sm" data-action="delete">Delete</button>
        </div>
      `;
      node.querySelector('[data-action="load"]').addEventListener('click', () => loadHistoryItem(item.id));
      node.querySelector('[data-action="delete"]').addEventListener('click', () => deleteHistoryItem(item.id));
      els.historyList.appendChild(node);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function openHistorySidebar() {
    els.historySidebar.classList.add('sidebar--open');
    els.sidebarOverlay.classList.add('sidebar-overlay--visible');
    els.historySidebar.setAttribute('aria-hidden', 'false');
  }

  function closeHistorySidebar() {
    els.historySidebar.classList.remove('sidebar--open');
    els.sidebarOverlay.classList.remove('sidebar-overlay--visible');
    els.historySidebar.setAttribute('aria-hidden', 'true');
  }

  function setupHistorySidebar() {
    $('historyToggle').addEventListener('click', openHistorySidebar);
    $('closeHistory').addEventListener('click', closeHistorySidebar);
    els.sidebarOverlay.addEventListener('click', closeHistorySidebar);

    $('btnExportHistory').addEventListener('click', () => {
      const history = loadHistory();
      if (!history.length) {
        showToast('No history to export', '⚠️');
        return;
      }
      downloadJSON(history, `prompt-history-${Date.now()}.json`);
      showToast('History exported');
    });

    $('btnClearHistory').addEventListener('click', () => {
      if (!loadHistory().length) return;
      if (!confirm('Clear all saved prompts? This cannot be undone.')) return;
      persistHistory([]);
      renderHistory();
      showToast('History cleared', '🗑️');
    });
  }

  // ---------- Reset ----------

  function resetAll() {
    els.productName.value = '';
    els.brandName.value = '';
    els.productCategory.value = '';
    els.targetAudience.value = '';
    els.keyFeatures.value = '';
    els.priceRange.value = '';

    setChipActive('toneGroup', 'Professional');
    setChipActive('lengthGroup', 'Medium (100 words)');
    setChipActive('styleGroup', 'Formal');
    setCta(true);
    state.tone = 'Professional';
    state.length = 'Medium (100 words)';
    state.style = 'Formal';
    clearTemplateSelection();

    const defaults = { temperature: 0.7, topK: 50, topP: 0.95, maxLength: 150, numVariations: 1 };
    Object.entries(defaults).forEach(([key, value]) => {
      params[key] = value;
      const input = $(key);
      const label = $(`${key}Value`);
      if (input) input.value = value;
      if (label) label.textContent = value;
    });

    els.pythonPanel.classList.add('hidden');
    updatePreview();
  }

  // ---------- Theme ----------

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    $('themeIconDark').classList.toggle('hidden', theme === 'light');
    $('themeIconLight').classList.toggle('hidden', theme === 'dark');
    localStorage.setItem(THEME_KEY, theme);
  }

  function setupTheme() {
    const saved = localStorage.getItem(THEME_KEY) || 'dark';
    applyTheme(saved);
    $('themeToggle').addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      applyTheme(current === 'dark' ? 'light' : 'dark');
    });
  }

  // ---------- Particle background ----------

  function setupParticles() {
    const canvas = $('particleCanvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    let width, height;

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    function makeParticles() {
      const count = Math.min(70, Math.floor((width * height) / 22000));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.8 + 0.6,
      }));
    }

    function tick() {
      ctx.clearRect(0, 0, width, height);
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      ctx.fillStyle = isLight ? 'rgba(139, 92, 246, 0.35)' : 'rgba(139, 92, 246, 0.55)';
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > width) p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      });
      requestAnimationFrame(tick);
    }

    window.addEventListener('resize', () => {
      resize();
      makeParticles();
    });

    resize();
    makeParticles();
    requestAnimationFrame(tick);
  }

  // ---------- Init ----------

  function init() {
    setupTheme();
    setupParticles();
    setupTemplates();
    setupChipGroup('toneGroup', 'tone');
    setupChipGroup('lengthGroup', 'length');
    setupChipGroup('styleGroup', 'style');
    setupCtaToggle();
    setupAdvancedPanel();
    setupFormFields();
    setupActions();
    setupHistorySidebar();
    renderHistory();
    updatePreview();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
