/**
 * Corrective RAG (CRAG) Agent - Frontend Application Logic
 * Pure JavaScript (ES6+) with Server-Sent Events (SSE) streaming.
 */

document.addEventListener('DOMContentLoaded', () => {
  // =========================================================================
  // DOM Element Selectors
  // =========================================================================
  
  // Sidebar & Configuration
  const appSidebar = document.getElementById('appSidebar');
  const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
  const sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');
  const openSettingsBtn = document.getElementById('openSettingsBtn');
  const saveConfigBtn = document.getElementById('saveConfigBtn');
  
  const nvidiaApiKeyInput = document.getElementById('nvidiaApiKey');
  const separateKeysToggle = document.getElementById('separateKeysToggle');
  const separateKeysSection = document.getElementById('separateKeysSection');
  const nvidiaChatApiKeyInput = document.getElementById('nvidiaChatApiKey');
  const nvidiaEmbeddingApiKeyInput = document.getElementById('nvidiaEmbeddingApiKey');
  const tavilyApiKeyInput = document.getElementById('tavilyApiKey');
  const voyageApiKeyInput = document.getElementById('voyageApiKey');
  const qdrantUrlInput = document.getElementById('qdrantUrl');
  const qdrantApiKeyInput = document.getElementById('qdrantApiKey');
  const chatModelSelect = document.getElementById('chatModelSelect');
  const customChatModelWrapper = document.getElementById('customChatModelWrapper');
  const customChatModelInput = document.getElementById('customChatModelInput');
  const embedModelSelect = document.getElementById('embedModelSelect');
  const customEmbedModelWrapper = document.getElementById('customEmbedModelWrapper');
  const customEmbedModelInput = document.getElementById('customEmbedModelInput');
  const chunkSizeInput = document.getElementById('chunkSize');
  const chunkSizeVal = document.getElementById('chunkSizeVal');
  const chunkOverlapInput = document.getElementById('chunkOverlap');
  const chunkOverlapVal = document.getElementById('chunkOverlapVal');

  // Header & Status
  const activeDocBadge = document.getElementById('activeDocBadge');
  const activeDocText = document.getElementById('activeDocText');
  const systemStatusPill = document.getElementById('systemStatusPill');
  const systemStatusText = document.getElementById('systemStatusText');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const apiKeyWarningBanner = document.getElementById('apiKeyWarningBanner');

  // Ingestion Elements
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const docUrlInput = document.getElementById('docUrlInput');
  const ingestUrlBtn = document.getElementById('ingestUrlBtn');
  const sampleChips = document.querySelectorAll('.chip-sample');
  
  const fileDropzone = document.getElementById('fileDropzone');
  const fileInput = document.getElementById('fileInput');
  const selectedFilePreview = document.getElementById('selectedFilePreview');
  const selectedFileName = document.getElementById('selectedFileName');
  const selectedFileSize = document.getElementById('selectedFileSize');
  const ingestFileBtn = document.getElementById('ingestFileBtn');

  const ingestionResultBox = document.getElementById('ingestionResultBox');
  const closeResultBtn = document.getElementById('closeResultBtn');
  const metricSource = document.getElementById('metricSource');
  const metricDocCount = document.getElementById('metricDocCount');
  const metricChunkCount = document.getElementById('metricChunkCount');
  const metricVectorDim = document.getElementById('metricVectorDim');
  const metricCollection = document.getElementById('metricCollection');

  // Query & Execution Elements
  const queryInput = document.getElementById('queryInput');
  const submitQueryBtn = document.getElementById('submitQueryBtn');
  const queryChips = document.querySelectorAll('.query-chip');
  const executionLog = document.getElementById('executionLog');
  const emptyLogState = document.getElementById('emptyLogState');
  const finalAnswerContainer = document.getElementById('finalAnswerContainer');
  const answerMarkdownBody = document.getElementById('answerMarkdownBody');
  const copyAnswerBtn = document.getElementById('copyAnswerBtn');

  // Toast Container
  const toastContainer = document.getElementById('toastContainer');

  // State
  let currentEventSource = null;
  let selectedUploadFile = null;
  let rawFinalAnswer = '';

  // Pipeline Step Nodes
  const stepNodes = {
    retrieve: document.getElementById('step-retrieve'),
    grade_documents: document.getElementById('step-grade_documents'),
    transform_query: document.getElementById('step-transform_query'),
    web_search: document.getElementById('step-web_search'),
    generate: document.getElementById('step-generate'),
  };

  const stepBadges = {
    retrieve: document.getElementById('badge-retrieve'),
    grade_documents: document.getElementById('badge-grade_documents'),
    transform_query: document.getElementById('badge-transform_query'),
    web_search: document.getElementById('badge-web_search'),
    generate: document.getElementById('badge-generate'),
  };

  // =========================================================================
  // Initialization & Configuration
  // =========================================================================

  async function loadConfig() {
    try {
      const response = await fetch('/api/config');
      if (!response.ok) throw new Error('Failed to fetch configuration');
      const data = await response.json();

      nvidiaApiKeyInput.value = data.nvidia_api_key || '';
      nvidiaChatApiKeyInput.value = data.nvidia_chat_api_key || '';
      nvidiaEmbeddingApiKeyInput.value = data.nvidia_embedding_api_key || '';

      if (data.nvidia_chat_api_key || data.nvidia_embedding_api_key) {
        separateKeysToggle.checked = true;
        separateKeysSection.style.display = 'flex';
      } else {
        separateKeysToggle.checked = false;
        separateKeysSection.style.display = 'none';
      }

      tavilyApiKeyInput.value = data.tavily_api_key || '';
      if (voyageApiKeyInput) {
        voyageApiKeyInput.value = data.voyage_api_key || '';
      }
      qdrantUrlInput.value = data.qdrant_url || 'http://localhost:6333';
      qdrantApiKeyInput.value = data.qdrant_api_key || '';

      // Populate Chat Model Select Options
      if (data.available_chat_models && data.available_chat_models.length) {
        chatModelSelect.innerHTML = '';
        let chatModelFound = false;
        data.available_chat_models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          if (m === data.nvidia_chat_model) {
            opt.selected = true;
            chatModelFound = true;
          }
          chatModelSelect.appendChild(opt);
        });

        // Append Custom Option
        const customOpt = document.createElement('option');
        customOpt.value = '__custom__';
        customOpt.textContent = '✨ Custom Model ID...';
        chatModelSelect.appendChild(customOpt);

        if (!chatModelFound && data.nvidia_chat_model) {
          customOpt.selected = true;
          customChatModelWrapper.style.display = 'block';
          customChatModelInput.value = data.nvidia_chat_model;
        } else {
          customChatModelWrapper.style.display = 'none';
        }
      }

      // Populate Embedding Model Select Options
      if (data.available_embedding_models && data.available_embedding_models.length) {
        embedModelSelect.innerHTML = '';
        let embedModelFound = false;
        data.available_embedding_models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m;
          opt.textContent = m;
          if (m === data.nvidia_embedding_model) {
            opt.selected = true;
            embedModelFound = true;
          }
          embedModelSelect.appendChild(opt);
        });

        // Append Custom Option
        const customOpt = document.createElement('option');
        customOpt.value = '__custom__';
        customOpt.textContent = '✨ Custom Model ID...';
        embedModelSelect.appendChild(customOpt);

        if (!embedModelFound && data.nvidia_embedding_model) {
          customOpt.selected = true;
          customEmbedModelWrapper.style.display = 'block';
          customEmbedModelInput.value = data.nvidia_embedding_model;
        } else {
          customEmbedModelWrapper.style.display = 'none';
        }
      }

      if (data.chunk_size) {
        chunkSizeInput.value = data.chunk_size;
        chunkSizeVal.textContent = data.chunk_size;
      }
      if (data.chunk_overlap) {
        chunkOverlapInput.value = data.chunk_overlap;
        chunkOverlapVal.textContent = data.chunk_overlap;
      }

      if (data.default_doc_url && !docUrlInput.value) {
        docUrlInput.value = data.default_doc_url;
      }

      updateIngestedSourceBadge(data.ingested_source);
      updateApiKeyWarning(data.has_nvidia_key);

    } catch (err) {
      console.error('Config fetch error:', err);
      showToast('Could not load configuration from server.', 'error');
    }
  }

  async function saveConfig() {
    let selectedChatModel = chatModelSelect.value;
    if (selectedChatModel === '__custom__') {
      selectedChatModel = customChatModelInput.value.trim();
      if (!selectedChatModel) {
        showToast('Please enter a valid Custom Chat Model ID.', 'warning');
        customChatModelInput.focus();
        return;
      }
    }

    let selectedEmbedModel = embedModelSelect.value;
    if (selectedEmbedModel === '__custom__') {
      selectedEmbedModel = customEmbedModelInput.value.trim();
      if (!selectedEmbedModel) {
        showToast('Please enter a valid Custom Embedding Model ID.', 'warning');
        customEmbedModelInput.focus();
        return;
      }
    }

    saveConfigBtn.disabled = true;
    saveConfigBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

    const payload = {
      nvidia_api_key: nvidiaApiKeyInput.value.trim(),
      nvidia_chat_api_key: separateKeysToggle.checked ? nvidiaChatApiKeyInput.value.trim() : '',
      nvidia_embedding_api_key: separateKeysToggle.checked ? nvidiaEmbeddingApiKeyInput.value.trim() : '',
      voyage_api_key: voyageApiKeyInput ? voyageApiKeyInput.value.trim() : '',
      tavily_api_key: tavilyApiKeyInput.value.trim(),
      qdrant_url: qdrantUrlInput.value.trim(),
      qdrant_api_key: qdrantApiKeyInput.value.trim(),
      nvidia_chat_model: selectedChatModel,
      nvidia_embedding_model: selectedEmbedModel,
      chunk_size: parseInt(chunkSizeInput.value, 10),
      chunk_overlap: parseInt(chunkOverlapInput.value, 10),
    };

    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await res.json();
      if (res.ok) {
        showToast('Settings saved successfully!', 'success');
        updateApiKeyWarning(result.has_nvidia_key);
        if (window.innerWidth <= 900) closeSidebar();
      } else {
        showToast(result.detail || 'Failed to save settings.', 'error');
      }
    } catch (err) {
      console.error('Config save error:', err);
      showToast('Error saving configuration.', 'error');
    } finally {
      saveConfigBtn.disabled = false;
      saveConfigBtn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> <span>Save Settings</span>';
    }
  }

  function updateApiKeyWarning(hasKey) {
    if (hasKey) {
      apiKeyWarningBanner.style.display = 'none';
      setSystemStatus('Ready', 'ready');
    } else {
      apiKeyWarningBanner.style.display = 'flex';
      setSystemStatus('API Key Needed', 'error');
    }
  }

  function updateIngestedSourceBadge(source) {
    if (source) {
      activeDocText.textContent = source;
      activeDocBadge.classList.add('has-doc');
      activeDocBadge.title = `Currently indexed: ${source}`;
    } else {
      activeDocText.textContent = 'No document indexed';
      activeDocBadge.classList.remove('has-doc');
      activeDocBadge.title = 'No document indexed yet';
    }
  }

  function setSystemStatus(text, type = 'ready') {
    systemStatusText.textContent = text;
    systemStatusPill.className = `status-pill status-${type}`;
  }

  // =========================================================================
  // Sidebar & Theme Handlers
  // =========================================================================

  function openSidebar() {
    appSidebar.classList.add('open');
    sidebarBackdrop.classList.add('active');
  }

  function closeSidebar() {
    appSidebar.classList.remove('open');
    sidebarBackdrop.classList.remove('active');
  }

  sidebarToggleBtn.addEventListener('click', () => {
    appSidebar.classList.contains('open') ? closeSidebar() : openSidebar();
  });

  sidebarCloseBtn?.addEventListener('click', closeSidebar);
  sidebarBackdrop?.addEventListener('click', closeSidebar);
  openSettingsBtn?.addEventListener('click', openSidebar);
  saveConfigBtn.addEventListener('click', saveConfig);

  // Separate API Keys Toggle
  separateKeysToggle.addEventListener('change', () => {
    if (separateKeysToggle.checked) {
      separateKeysSection.style.display = 'flex';
    } else {
      separateKeysSection.style.display = 'none';
    }
  });

  // Custom Model Dropdown Toggles
  chatModelSelect.addEventListener('change', () => {
    if (chatModelSelect.value === '__custom__') {
      customChatModelWrapper.style.display = 'block';
      customChatModelInput.focus();
    } else {
      customChatModelWrapper.style.display = 'none';
    }
  });

  embedModelSelect.addEventListener('change', () => {
    if (embedModelSelect.value === '__custom__') {
      customEmbedModelWrapper.style.display = 'block';
      customEmbedModelInput.focus();
    } else {
      customEmbedModelWrapper.style.display = 'none';
    }

    const isVoyage = embedModelSelect.value.toLowerCase().includes('voyage');
    const voyageGroup = document.getElementById('voyageApiKeyGroup');
    if (voyageGroup) {
      if (isVoyage) {
        voyageGroup.style.background = 'rgba(99, 102, 241, 0.08)';
        voyageGroup.style.borderRadius = '8px';
        voyageGroup.style.padding = '8px 12px';
        voyageGroup.style.border = '1px solid var(--accent-primary, #6366f1)';
      } else {
        voyageGroup.style.background = 'transparent';
        voyageGroup.style.borderRadius = '0';
        voyageGroup.style.padding = '0';
        voyageGroup.style.border = 'none';
      }
    }
  });

  // Sliders synchronization
  chunkSizeInput.addEventListener('input', (e) => {
    chunkSizeVal.textContent = e.target.value;
  });
  chunkOverlapInput.addEventListener('input', (e) => {
    chunkOverlapVal.textContent = e.target.value;
  });

  // Password Reveal Toggle
  document.querySelectorAll('.btn-toggle-pw').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      const input = document.getElementById(targetId);
      const icon = btn.querySelector('i');
      if (input.type === 'password') {
        input.type = 'text';
        icon.classList.replace('fa-eye', 'fa-eye-slash');
      } else {
        input.type = 'password';
        icon.classList.replace('fa-eye-slash', 'fa-eye');
      }
    });
  });

  // Theme Toggle (Dark / Light)
  const savedTheme = localStorage.getItem('crag_theme') || 'theme-dark';
  document.body.className = savedTheme;
  updateThemeIcon();

  themeToggleBtn.addEventListener('click', () => {
    if (document.body.classList.contains('theme-dark')) {
      document.body.classList.replace('theme-dark', 'theme-light');
      localStorage.setItem('crag_theme', 'theme-light');
    } else {
      document.body.classList.replace('theme-light', 'theme-dark');
      localStorage.setItem('crag_theme', 'theme-dark');
    }
    updateThemeIcon();
  });

  function updateThemeIcon() {
    const isDark = document.body.classList.contains('theme-dark');
    themeToggleBtn.innerHTML = isDark ? '<i class="fa-solid fa-moon"></i>' : '<i class="fa-solid fa-sun"></i>';
  }

  // =========================================================================
  // Document Ingestion Logic
  // =========================================================================

  // Ingestion Tabs
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
    });
  });

  // Example URL Chips
  sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const url = chip.getAttribute('data-url');
      if (url) docUrlInput.value = url;
    });
  });

  // File Dropzone Interaction
  fileDropzone.addEventListener('click', () => fileInput.click());

  fileDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileDropzone.classList.add('dragover');
  });

  fileDropzone.addEventListener('dragleave', () => {
    fileDropzone.classList.remove('dragover');
  });

  fileDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    fileDropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedUploadFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);
    selectedFilePreview.style.display = 'flex';

    // Update icon based on file extension
    const ext = file.name.split('.').pop().toLowerCase();
    const iconEl = document.getElementById('fileTypeIcon');
    if (ext === 'pdf') {
      iconEl.className = 'fa-solid fa-file-pdf file-icon';
    } else if (ext === 'txt') {
      iconEl.className = 'fa-solid fa-file-lines file-icon';
    } else if (ext === 'md') {
      iconEl.className = 'fa-brands fa-markdown file-icon';
    } else {
      iconEl.className = 'fa-solid fa-file file-icon';
    }
  }

  function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
  }

  // URL Ingest Action
  ingestUrlBtn.addEventListener('click', async () => {
    const url = docUrlInput.value.trim();
    if (!url) {
      showToast('Please enter a valid document URL.', 'warning');
      return;
    }

    setIngestionLoading(true, ingestUrlBtn, 'Ingesting URL...');
    try {
      const res = await fetch('/api/ingest/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();

      if (res.ok && data.status === 'success') {
        showIngestionResult(data);
        updateIngestedSourceBadge(data.source);
        showToast('Document successfully indexed into Qdrant!', 'success');
      } else {
        const errorMsg = data.error || data.detail || 'Ingestion failed';
        showToast(errorMsg, 'error');
      }
    } catch (err) {
      console.error('URL ingestion error:', err);
      showToast('Error during document ingestion.', 'error');
    } finally {
      setIngestionLoading(false, ingestUrlBtn, 'Ingest URL');
    }
  });

  // File Ingest Action
  ingestFileBtn.addEventListener('click', async () => {
    if (!selectedUploadFile) {
      showToast('Please select a file first.', 'warning');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedUploadFile);

    setIngestionLoading(true, ingestFileBtn, 'Ingesting File...');
    try {
      const res = await fetch('/api/ingest/file', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (res.ok && data.status === 'success') {
        showIngestionResult(data);
        updateIngestedSourceBadge(data.source);
        showToast(`File ${data.source} indexed successfully!`, 'success');
      } else {
        const errorMsg = data.error || data.detail || 'File ingestion failed';
        showToast(errorMsg, 'error');
      }
    } catch (err) {
      console.error('File ingestion error:', err);
      showToast('Error during file ingestion.', 'error');
    } finally {
      setIngestionLoading(false, ingestFileBtn, 'Ingest File');
    }
  });

  function setIngestionLoading(isLoading, button, text) {
    if (isLoading) {
      button.disabled = true;
      button.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>${text}</span>`;
      setSystemStatus('Ingesting Document...', 'running');
    } else {
      button.disabled = false;
      button.innerHTML = `<i class="fa-solid fa-bolt"></i> <span>${text}</span>`;
      setSystemStatus('Ready', 'ready');
    }
  }

  function showIngestionResult(res) {
    metricSource.textContent = res.source;
    metricDocCount.textContent = res.doc_count;
    metricChunkCount.textContent = res.chunk_count;
    metricVectorDim.textContent = res.vector_dim;
    metricCollection.textContent = res.collection_name;

    ingestionResultBox.style.display = 'block';
  }

  closeResultBtn?.addEventListener('click', () => {
    ingestionResultBox.style.display = 'none';
  });

  // =========================================================================
  // Query Execution & LangGraph Stepper / SSE
  // =========================================================================

  // Suggested Query Chips
  queryChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      if (q) {
        queryInput.value = q;
        submitQuery();
      }
    });
  });

  // Enter Key on Query Input
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitQuery();
    }
  });

  submitQueryBtn.addEventListener('click', submitQuery);

  function resetExecutionUI() {
    // Reset Stepper nodes
    Object.keys(stepNodes).forEach(key => {
      const nodeEl = stepNodes[key];
      const badgeEl = stepBadges[key];
      if (nodeEl && badgeEl) {
        nodeEl.className = 'stepper-node';
        if (key === 'transform_query' || key === 'web_search') {
          badgeEl.textContent = 'Conditional';
        } else {
          badgeEl.textContent = 'Pending';
        }
      }
    });

    // Clear Execution Logs
    executionLog.innerHTML = '';
    finalAnswerContainer.style.display = 'none';
    answerMarkdownBody.innerHTML = '';
    rawFinalAnswer = '';
  }

  function submitQuery() {
    const question = queryInput.value.trim();
    if (!question) {
      showToast('Please enter a question to execute.', 'warning');
      return;
    }

    if (currentEventSource) {
      currentEventSource.close();
    }

    resetExecutionUI();
    setQueryRunningState(true);

    // Initial node activation
    setNodeStatus('retrieve', 'active', 'Running...');

    const streamUrl = `/api/query/stream?question=${encodeURIComponent(question)}`;
    currentEventSource = new EventSource(streamUrl);

    currentEventSource.addEventListener('step', (e) => {
      try {
        const payload = JSON.parse(e.data);
        handleStepEvent(payload.node, payload.state);
      } catch (err) {
        console.error('Step parse error:', err);
      }
    });

    currentEventSource.addEventListener('answer', (e) => {
      try {
        const payload = JSON.parse(e.data);
        handleAnswerEvent(payload.generation);
      } catch (err) {
        console.error('Answer parse error:', err);
      }
    });

    currentEventSource.addEventListener('done', () => {
      completeQueryExecution();
    });

    currentEventSource.addEventListener('error', (e) => {
      console.error('SSE Error:', e);
      try {
        if (e.data) {
          const errData = JSON.parse(e.data);
          showToast(errData.error || 'Execution error encountered.', 'error');
        } else {
          showToast('Connection to server stream interrupted.', 'error');
        }
      } catch {
        showToast('Stream execution stopped.', 'error');
      }
      completeQueryExecution();
    });
  }

  function setNodeStatus(nodeName, status, badgeText) {
    const nodeEl = stepNodes[nodeName];
    const badgeEl = stepBadges[nodeName];
    if (nodeEl && badgeEl) {
      nodeEl.className = `stepper-node ${status}`;
      badgeEl.textContent = badgeText || (status === 'completed' ? 'Done' : status);
    }
  }

  function handleStepEvent(nodeName, state) {
    setNodeStatus(nodeName, 'completed', 'Completed');

    // Prepare next node's active state
    if (nodeName === 'retrieve') {
      setNodeStatus('grade_documents', 'active', 'Running...');
    } else if (nodeName === 'grade_documents') {
      const runWebSearch = state.run_web_search === 'Yes' || state.run_web_search === true;
      if (runWebSearch) {
        setNodeStatus('transform_query', 'active', 'Running...');
      } else {
        setNodeStatus('transform_query', 'skipped', 'Skipped');
        setNodeStatus('web_search', 'skipped', 'Skipped');
        setNodeStatus('generate', 'active', 'Running...');
      }
    } else if (nodeName === 'transform_query') {
      setNodeStatus('web_search', 'active', 'Running...');
    } else if (nodeName === 'web_search') {
      setNodeStatus('generate', 'active', 'Running...');
    }

    // Render Execution Step Accordion Card
    renderStepAccordion(nodeName, state);
  }

  function renderStepAccordion(nodeName, state) {
    if (emptyLogState) emptyLogState.remove();

    const nodeIcons = {
      retrieve: 'fa-database',
      grade_documents: 'fa-scale-balanced',
      transform_query: 'fa-wand-magic-sparkles',
      web_search: 'fa-earth-americas',
      generate: 'fa-sparkles',
    };

    const nodeLabels = {
      retrieve: 'Document Retrieval',
      grade_documents: 'Document Relevance Grading',
      transform_query: 'Query Optimization & Transformation',
      web_search: 'Tavily Fallback Web Search',
      generate: 'Response Generation',
    };

    const icon = nodeIcons[nodeName] || 'fa-gear';
    const label = nodeLabels[nodeName] || nodeName;
    const isAutoOpen = nodeName === 'generate' || nodeName === 'grade_documents';

    const accordion = document.createElement('div');
    accordion.className = `step-accordion ${isAutoOpen ? 'open' : ''}`;
    accordion.id = `accordion-${nodeName}`;

    // Custom inner details per node
    let contentHtml = '';

    if (nodeName === 'retrieve' || nodeName === 'web_search') {
      const docs = state.documents || [];
      if (docs.length > 0) {
        contentHtml += `<p class="input-hint" style="margin-bottom: 10px;">Retrieved <strong>${docs.length}</strong> context chunk(s):</p>`;
        contentHtml += '<div class="doc-card-list">';
        docs.forEach((d, idx) => {
          const src = d.source || 'Unknown';
          const title = d.title || 'Document';
          const text = d.snippet || d.page_content || '';
          contentHtml += `
            <div class="doc-card">
              <div class="doc-card-header">
                <strong>#${idx + 1} ${title}</strong>
                <span class="doc-source-badge">${src}</span>
              </div>
              <div class="doc-snippet">${escapeHtml(text)}</div>
            </div>
          `;
        });
        contentHtml += '</div>';
      } else {
        contentHtml += '<p class="input-hint">No documents found.</p>';
      }
    } else if (nodeName === 'grade_documents') {
      const isSearchNeeded = state.run_web_search === 'Yes' || state.run_web_search === true;
      const docs = state.documents || [];
      contentHtml += `
        <div style="display: flex; gap: 14px; margin-bottom: 12px; flex-wrap: wrap;">
          <div class="status-pill status-${isSearchNeeded ? 'running' : 'ready'}">
            <span>Web Search Fallback: <strong>${isSearchNeeded ? 'Required (Yes)' : 'Not Needed (No)'}</strong></span>
          </div>
          <div class="status-pill">
            <span>Relevant Docs Retained: <strong>${docs.length}</strong></span>
          </div>
        </div>
      `;
    } else if (nodeName === 'transform_query') {
      contentHtml += `
        <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.88rem;">
          <div><span class="text-muted">Original Query:</span> <strong>${escapeHtml(state.original_question || state.question || '')}</strong></div>
          <div><span class="text-muted">Optimized Search Query:</span> <code style="color: var(--accent-secondary); font-size: 0.95em;">${escapeHtml(state.question || '')}</code></div>
        </div>
      `;
    } else if (nodeName === 'generate') {
      contentHtml += `<p class="input-hint">Answer synthesized from ${state.documents?.length || 0} context sources.</p>`;
    }

    // Append Raw State Inspector inside accordion
    contentHtml += `
      <details style="margin-top: 14px;">
        <summary style="font-size: 0.78rem; color: var(--text-muted); cursor: pointer;">🔍 View Raw Node State Payload</summary>
        <pre class="raw-state-box" style="margin-top: 8px;"><code>${escapeHtml(JSON.stringify(state, null, 2))}</code></pre>
      </details>
    `;

    accordion.innerHTML = `
      <div class="accordion-header">
        <div class="accordion-title">
          <i class="fa-solid ${icon} text-accent"></i>
          <span>${label}</span>
          <span class="node-chip">${nodeName}</span>
        </div>
        <i class="fa-solid fa-chevron-down accordion-icon"></i>
      </div>
      <div class="accordion-body">
        ${contentHtml}
      </div>
    `;

    // Toggle click handler
    accordion.querySelector('.accordion-header').addEventListener('click', () => {
      accordion.classList.toggle('open');
    });

    executionLog.appendChild(accordion);
  }

  function handleAnswerEvent(generation) {
    if (!generation) return;
    rawFinalAnswer = generation;

    // Render markdown with marked.js
    if (typeof marked !== 'undefined' && marked.parse) {
      answerMarkdownBody.innerHTML = marked.parse(generation);
    } else {
      answerMarkdownBody.innerHTML = `<pre style="white-space: pre-wrap;">${escapeHtml(generation)}</pre>`;
    }

    finalAnswerContainer.style.display = 'block';

    // Smooth scroll down to answer
    finalAnswerContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function completeQueryExecution() {
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
    setQueryRunningState(false);
  }

  function setQueryRunningState(isRunning) {
    if (isRunning) {
      submitQueryBtn.disabled = true;
      submitQueryBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Running CRAG...</span>';
      setSystemStatus('Executing Graph...', 'running');
    } else {
      submitQueryBtn.disabled = false;
      submitQueryBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> <span>Execute CRAG</span>';
      setSystemStatus('Ready', 'ready');
    }
  }

  // Copy Final Answer
  copyAnswerBtn.addEventListener('click', async () => {
    if (!rawFinalAnswer) return;
    try {
      await navigator.clipboard.writeText(rawFinalAnswer);
      copyAnswerBtn.innerHTML = '<i class="fa-solid fa-check text-success"></i> <span>Copied!</span>';
      setTimeout(() => {
        copyAnswerBtn.innerHTML = '<i class="fa-regular fa-copy"></i> <span>Copy</span>';
      }, 2000);
      showToast('Answer copied to clipboard!', 'info');
    } catch {
      showToast('Failed to copy to clipboard.', 'error');
    }
  });

  // =========================================================================
  // Toast System
  // =========================================================================

  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const iconMap = {
      success: 'fa-circle-check',
      error: 'fa-circle-exclamation',
      warning: 'fa-triangle-exclamation',
      info: 'fa-circle-info',
    };

    const icon = iconMap[type] || iconMap.info;

    toast.innerHTML = `
      <i class="fa-solid ${icon} toast-icon"></i>
      <span class="toast-message">${escapeHtml(message)}</span>
    `;

    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(16px)';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return str;
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // =========================================================================
  // Start Application
  // =========================================================================
  loadConfig();
});
