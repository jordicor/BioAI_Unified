// ============================================
// BIOAI UNIFIED STREAM MONITOR
// Direct connection (no proxy needed)
// ============================================

// Direct connection - no proxy needed
const STREAM_BASE = '';
const PHASES = ['generation', 'qa', 'preflight', 'consensus', 'gransabio', 'status', 'everything'];
// Phases that can be hard-switched (excluded from subscription)
const HARD_SWITCHABLE_PHASES = ['generation', 'qa', 'preflight', 'consensus', 'gransabio', 'status'];

// State
let eventSource = null;
let currentProjectId = null;
let stats = {};
let lastProjectData = null;  // Cache for re-rendering dashboard

// Dual-mode support: 'project' or 'session'
let currentConnectionMode = 'project';
let currentSessionId = null;

// Helper to check if we're in Overview mode
function isOverviewMode() {
    return viewState.selectedRequestId === null;
}

// Panel toggle states: 'on' | 'soft' | 'hard'
// 'everything' panel only supports 'on' | 'soft' (no hard switch)
const panelStates = {};
PHASES.forEach(phase => {
    panelStates[phase] = 'on';
});

// Initialize stats
PHASES.forEach(phase => {
    stats[phase] = { chunks: 0, bytes: 0 };
});

// ============================================
// REQUEST TRACKING (Phase 1)
// ============================================

// Request data storage
const requestsData = {
    orderedRequestIds: [],  // Maintains order of arrival
    requests: {}            // Keyed by session_id
};

// View state for request selection
const viewState = {
    selectedRequestId: null,
    autoFollow: true        // Auto-select new requests when they arrive
};

// Maximum visible tabs before overflow
const MAX_VISIBLE_TABS = 6;

// ============================================
// PHASE TOGGLE MANAGEMENT
// ============================================

function initToggleListeners() {
    document.querySelectorAll('.phase-toggle').forEach(toggle => {
        const phase = toggle.dataset.phase;
        toggle.querySelectorAll('.toggle-segment').forEach(segment => {
            segment.addEventListener('click', () => {
                const newState = segment.dataset.state;
                setPhaseState(phase, newState);
            });
        });
    });
}

function setPhaseState(phase, newState) {
    const oldState = panelStates[phase];

    // No change
    if (oldState === newState) return;

    panelStates[phase] = newState;
    updateToggleUI(phase, newState);
    updatePanelVisual(phase, newState);

    log(`Phase '${phase}' switched: ${oldState} -> ${newState}`, 'info');

    // Hard switch requires reconnection (but not for 'everything' panel)
    if (phase !== 'everything' && (newState === 'hard' || oldState === 'hard')) {
        if (eventSource && currentProjectId) {
            log('Hard switch detected - reconnecting with updated phases...', 'warn');
            reconnectWithActivePhases();
        }
    }
}

function updateToggleUI(phase, state) {
    const toggle = document.querySelector(`.phase-toggle[data-phase="${phase}"]`);
    if (!toggle) return;

    toggle.querySelectorAll('.toggle-segment').forEach(segment => {
        segment.classList.remove('active');
        if (segment.dataset.state === state) {
            segment.classList.add('active');
        }
    });
}

function updatePanelVisual(phase, state) {
    const panel = document.querySelector(`.stream-panel[data-phase="${phase}"]`);
    if (!panel) return;

    panel.classList.remove('muted-soft', 'muted-hard');
    if (state === 'soft') {
        panel.classList.add('muted-soft');
    } else if (state === 'hard') {
        panel.classList.add('muted-hard');
    }
}

function getActivePhases() {
    // Return phases that are NOT in 'hard' state
    return HARD_SWITCHABLE_PHASES.filter(phase => panelStates[phase] !== 'hard');
}

function reconnectWithActivePhases() {
    if (!currentProjectId) return;

    const activePhases = getActivePhases();

    if (activePhases.length === 0) {
        log('All phases are hard-muted. Disconnecting...', 'warn');
        disconnect();
        return;
    }

    // Close current connection
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    // Reconnect with only active phases
    const phasesParam = activePhases.join(',');
    const streamUrl = `${STREAM_BASE}/stream/project/${currentProjectId}?phases=${phasesParam}`;

    log(`Reconnecting with phases: ${phasesParam}`, 'info');
    setConnectionStatus('connecting', 'RECONNECTING...');

    try {
        eventSource = new EventSource(streamUrl);

        eventSource.onopen = function() {
            log('Reconnected successfully', 'success');
            setConnectionStatus('connected', 'ONLINE');
            // Update badges for active phases only
            PHASES.forEach(phase => {
                if (panelStates[phase] === 'hard') {
                    setBadge(phase, 'muted');
                } else {
                    setBadge(phase, 'active');
                }
            });
        };

        eventSource.onmessage = function(event) {
            handleMessage(event.data);
        };

        eventSource.onerror = function(error) {
            handleConnectionError(error);
        };

    } catch (error) {
        log(`Reconnection failed: ${error.message}`, 'error');
        setConnectionStatus('disconnected', 'ERROR');
    }
}

function handleConnectionError(error) {
    console.error('EventSource error:', error);

    if (eventSource.readyState === EventSource.CLOSED) {
        log('Connection closed by server', 'warn');
        setConnectionStatus('disconnected', 'CLOSED');
        PHASES.forEach(phase => setBadge(phase, 'idle'));
    } else if (eventSource.readyState === EventSource.CONNECTING) {
        log('Reconnecting...', 'warn');
        setConnectionStatus('connecting', 'RECONNECTING...');
    } else {
        log('Connection error', 'error');
        setConnectionStatus('disconnected', 'ERROR');
    }
}

function resetAllTogglesToOn() {
    PHASES.forEach(phase => {
        panelStates[phase] = 'on';
        updateToggleUI(phase, 'on');
        updatePanelVisual(phase, 'on');
    });
}

// ============================================
// LOGGING
// ============================================

function log(message, type = 'info') {
    const logContent = document.getElementById('logContent');
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="timestamp">[${timestamp}]</span>${escapeHtml(message)}`;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;

    // Also log to console for debugging
    console.log(`[${type.toUpperCase()}] ${message}`);
}

function clearLog() {
    document.getElementById('logContent').innerHTML = '';
    log('Log cleared', 'info');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// UI UPDATES
// ============================================

function setConnectionStatus(status, message) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const connectBtn = document.getElementById('connectBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');

    indicator.className = `status-indicator ${status}`;
    statusText.textContent = message;

    if (status === 'connected') {
        connectBtn.style.display = 'none';
        disconnectBtn.style.display = 'inline-block';
    } else {
        connectBtn.style.display = 'inline-block';
        disconnectBtn.style.display = 'none';
        connectBtn.disabled = (status === 'connecting');
    }
}

function setBadge(phase, status) {
    const badge = document.getElementById(`badge-${phase}`);
    if (!badge) return;
    badge.textContent = status;
    badge.className = 'stream-badge';
    if (status === 'active' || status === 'live') {
        badge.classList.add('active');
    } else if (status === 'receiving') {
        badge.classList.add('receiving');
    }
}

function appendContent(phase, text) {
    const content = document.getElementById(`content-${phase}`);
    if (!content) return;

    // Check toggle state - soft/hard muted panels don't display content
    const state = panelStates[phase];
    if (state === 'soft' || state === 'hard') {
        // Still update stats for visibility, but don't append content
        stats[phase].chunks++;
        stats[phase].bytes += text.length;

        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = stats[phase].chunks;
        if (bytesEl) bytesEl.textContent = stats[phase].bytes;

        // Brief flash to show data is arriving but muted
        setBadge(phase, 'muted');
        return;
    }

    content.textContent += text;
    content.scrollTop = content.scrollHeight;

    // Update stats
    stats[phase].chunks++;
    stats[phase].bytes += text.length;

    const chunksEl = document.getElementById(`chunks-${phase}`);
    const bytesEl = document.getElementById(`bytes-${phase}`);
    if (chunksEl) chunksEl.textContent = stats[phase].chunks;
    if (bytesEl) bytesEl.textContent = stats[phase].bytes;

    // Show receiving state with pulse - debounced return to active
    showReceiving(phase);
}

// Debounce timers for each phase
const receivingTimers = {};

function showReceiving(phase) {
    const badge = document.getElementById(`badge-${phase}`);
    if (!badge) return;

    // Clear any existing timer for this phase
    if (receivingTimers[phase]) {
        clearTimeout(receivingTimers[phase]);
    }

    // Set to receiving with pulse effect
    badge.textContent = 'receiving';
    badge.className = 'stream-badge receiving pulse';

    // Debounced return to active - only after 400ms of no chunks
    receivingTimers[phase] = setTimeout(() => {
        badge.textContent = 'active';
        badge.className = 'stream-badge active';
    }, 400);
}

function clearAllContent() {
    // Clear request tracking structures
    requestsData.orderedRequestIds = [];
    requestsData.requests = {};
    viewState.selectedRequestId = null;  // Start in Overview mode
    viewState.autoFollow = true;
    lastProjectData = null;

    // Close overflow menu if open
    const overflowMenu = document.getElementById('tabsOverflowMenu');
    if (overflowMenu) overflowMenu.classList.add('hidden');

    // Clear phase panels and set overview mode
    PHASES.forEach(phase => {
        const content = document.getElementById(`content-${phase}`);
        if (content) {
            content.textContent = '';
            // Set overview mode flag for placeholder message
            if (phase !== 'status' && phase !== 'everything') {
                content.dataset.overviewMode = 'true';
            }
        }

        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = phase === 'status' || phase === 'everything' ? '0' : '-';
        if (bytesEl) bytesEl.textContent = phase === 'status' || phase === 'everything' ? '0' : '-';

        setBadge(phase, 'idle');
        stats[phase] = { chunks: 0, bytes: 0 };
    });

    // Render tabs (Overview will be shown)
    renderRequestTabs();

    // Hide status dashboard until data arrives
    hideDashboard();
}

// ============================================
// REQUEST PROCESSING
// ============================================

/**
 * Extract request info from event data and create/update request entry
 * @param {Object} data - Event data with session_id and request_name
 * @returns {Object|null} The request object, or null if no session_id
 */
function processRequestFromEvent(data) {
    const sessionId = data.session_id;
    const requestName = data.request_name || 'unknown';

    if (!sessionId) return null;

    // Create request if it doesn't exist
    if (!requestsData.requests[sessionId]) {
        requestsData.requests[sessionId] = {
            sessionId: sessionId,
            requestName: requestName,
            status: 'active',
            currentIteration: data.iteration || 1,
            maxIterations: data.max_iterations || 5,
            finalScore: null,
            // Content accumulated by phase
            content: {
                preflight: '',
                generation: '',
                qa: '',
                consensus: '',
                gransabio: ''
            },
            // Stats per phase
            stats: {
                preflight: { chunks: 0, bytes: 0 },
                generation: { chunks: 0, bytes: 0 },
                qa: { chunks: 0, bytes: 0 },
                consensus: { chunks: 0, bytes: 0 },
                gransabio: { chunks: 0, bytes: 0 }
            }
        };

        // Add to ordered list
        requestsData.orderedRequestIds.push(sessionId);

        // Auto-select if autoFollow is on (switch from Overview to this request)
        if (viewState.autoFollow) {
            viewState.selectedRequestId = sessionId;

            // Clear overview mode flag and prepare panels for content
            ['preflight', 'generation', 'qa', 'consensus', 'gransabio'].forEach(phase => {
                const contentEl = document.getElementById(`content-${phase}`);
                if (contentEl) {
                    delete contentEl.dataset.overviewMode;
                    contentEl.textContent = '';  // Start fresh
                }
            });

            // Update dashboard for this specific session
            renderProjectStatus(lastProjectData, sessionId);
        }

        // Trigger UI update
        renderRequestTabs();

        log(`New request detected: ${requestName} (${sessionId.slice(0, 8)}...)`, 'info');
    }

    return requestsData.requests[sessionId];
}

// ============================================
// REQUEST TABS UI
// ============================================

function renderRequestTabs() {
    const container = document.getElementById('requestTabsContainer');
    const panel = document.getElementById('requestTabsPanel');
    const infoBar = document.getElementById('requestInfoBar');

    if (!container) return;

    // Always show the tabs panel (Overview is always available)
    if (panel) panel.style.display = 'block';
    if (infoBar) infoBar.style.display = 'flex';

    // Clear container
    container.innerHTML = '';

    // Always add Overview tab first (fixed, not closable)
    const overviewTab = createOverviewTab();
    container.appendChild(overviewTab);

    const requests = requestsData.orderedRequestIds;

    // Adjust max visible: -1 because Overview occupies one slot
    const maxSessionTabs = MAX_VISIBLE_TABS - 1;
    const visibleCount = Math.min(requests.length, maxSessionTabs);
    const overflowCount = requests.length - visibleCount;

    // Render session tabs
    for (let i = 0; i < visibleCount; i++) {
        const sessionId = requests[i];
        const request = requestsData.requests[sessionId];
        const tab = createRequestTab(request);
        container.appendChild(tab);
    }

    // Add overflow button if needed
    if (overflowCount > 0) {
        const overflowBtn = document.createElement('button');
        overflowBtn.className = 'tabs-overflow-btn';
        overflowBtn.innerHTML = `+ ${overflowCount} more`;
        overflowBtn.onclick = (e) => {
            e.stopPropagation();
            toggleOverflowMenu();
        };
        container.appendChild(overflowBtn);

        // Render overflow menu items
        renderOverflowMenu(requests.slice(visibleCount));
    }

    // Update info bar
    updateRequestInfoBar();
}

function createRequestTab(request) {
    const tab = document.createElement('div');
    tab.className = 'request-tab';
    tab.dataset.sessionId = request.sessionId;
    tab.title = `${request.requestName}\nSession: ${request.sessionId}`;

    if (request.sessionId === viewState.selectedRequestId) {
        tab.classList.add('selected');
    }

    // Determine icon and status text
    let statusIcon, statusClass, infoText;
    switch (request.status) {
        case 'completed':
            statusIcon = '+';
            statusClass = 'completed';
            infoText = request.finalScore ? request.finalScore.toFixed(1) : 'done';
            break;
        case 'failed':
            statusIcon = 'x';
            statusClass = 'failed';
            infoText = 'failed';
            break;
        case 'pending':
            statusIcon = 'o';
            statusClass = 'pending';
            infoText = 'pending';
            break;
        default:  // active
            statusIcon = '*';
            statusClass = 'active';
            infoText = `iter ${request.currentIteration}/${request.maxIterations}`;
    }

    // Truncate name if needed
    const shortName = request.requestName.length > 14
        ? request.requestName.slice(0, 11) + '...'
        : request.requestName;

    tab.innerHTML = `
        <div class="tab-header">
            <span class="tab-status-icon ${statusClass}">[${statusIcon}]</span>
            <span class="tab-name">${escapeHtml(shortName)}</span>
        </div>
        <div class="tab-info">${infoText}</div>
        <div class="tab-close-zone">
            <span class="tab-close-btn" title="Close tab">&times;</span>
        </div>
    `;

    // Click on tab selects the request
    tab.onclick = (e) => {
        // Don't select if clicking on close button
        if (e.target.classList.contains('tab-close-btn')) return;
        selectRequest(request.sessionId);
    };

    // Click on close button closes the tab
    const closeBtn = tab.querySelector('.tab-close-btn');
    closeBtn.onclick = (e) => {
        e.stopPropagation();
        closeRequest(request.sessionId);
    };

    return tab;
}

/**
 * Create the special Overview tab (always first, not closable)
 */
function createOverviewTab() {
    const tab = document.createElement('div');
    tab.className = 'request-tab overview-tab';
    tab.title = 'View all sessions overview';

    if (isOverviewMode()) {
        tab.classList.add('selected');
    }

    // Count active sessions for display
    const activeCount = Object.values(requestsData.requests)
        .filter(r => r.status === 'active').length;
    const totalCount = requestsData.orderedRequestIds.length;

    tab.innerHTML = `
        <div class="tab-header">
            <span class="tab-status-icon overview">[#]</span>
            <span class="tab-name">Overview</span>
        </div>
        <div class="tab-info">${activeCount}/${totalCount} active</div>
    `;
    // No close-zone - Overview tab cannot be closed

    tab.onclick = () => selectOverview();

    return tab;
}

/**
 * Select Overview mode - show all sessions in dashboard, clear streaming panels
 */
function selectOverview() {
    viewState.selectedRequestId = null;
    viewState.autoFollow = false;  // User manually selected Overview

    // Update UI
    renderRequestTabs();

    // Clear streaming panels (show message instead)
    clearStreamingPanelsForOverview();

    // Render dashboard with ALL sessions (no filter)
    renderProjectStatus(lastProjectData, null);

    log('Switched to Overview mode', 'info');
}

/**
 * Clear streaming panels when in Overview mode
 */
function clearStreamingPanelsForOverview() {
    ['preflight', 'generation', 'qa', 'consensus', 'gransabio'].forEach(phase => {
        const contentEl = document.getElementById(`content-${phase}`);
        if (contentEl) {
            contentEl.textContent = '';
            contentEl.dataset.overviewMode = 'true';
        }

        // Show dashes for stats in overview mode
        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = '-';
        if (bytesEl) bytesEl.textContent = '-';

        setBadge(phase, 'idle');
    });
}

function selectRequest(sessionId) {
    viewState.selectedRequestId = sessionId;
    viewState.autoFollow = false;  // User manually selected, disable auto-follow

    // Update tabs UI
    renderRequestTabs();

    // Load streaming content for this session
    loadRequestContent(sessionId);

    // Clear overview mode flag from content elements
    ['preflight', 'generation', 'qa', 'consensus', 'gransabio'].forEach(phase => {
        const contentEl = document.getElementById(`content-${phase}`);
        if (contentEl) {
            delete contentEl.dataset.overviewMode;
        }
    });

    // Render dashboard filtered to only this session
    renderProjectStatus(lastProjectData, sessionId);

    const request = requestsData.requests[sessionId];
    log(`Switched to request: ${request?.requestName || sessionId}`, 'info');
}

function closeRequest(sessionId) {
    const request = requestsData.requests[sessionId];
    const requestName = request?.requestName || sessionId.slice(0, 8);

    // Remove from ordered list
    const index = requestsData.orderedRequestIds.indexOf(sessionId);
    if (index > -1) {
        requestsData.orderedRequestIds.splice(index, 1);
    }

    // Remove from requests object
    delete requestsData.requests[sessionId];

    // If this was the selected request, select another one
    if (viewState.selectedRequestId === sessionId) {
        if (requestsData.orderedRequestIds.length > 0) {
            // Select the previous tab, or the first one if we closed the first
            const newIndex = Math.max(0, index - 1);
            viewState.selectedRequestId = requestsData.orderedRequestIds[newIndex];
            loadRequestContent(viewState.selectedRequestId);
            renderProjectStatus(lastProjectData, viewState.selectedRequestId);
        } else {
            // No more requests - go back to Overview mode
            selectOverview();
            log(`Closed request: ${requestName}`, 'info');
            return;  // selectOverview already renders tabs
        }
    }

    // Re-render tabs
    renderRequestTabs();

    log(`Closed request: ${requestName}`, 'info');
}

function loadRequestContent(sessionId) {
    const request = requestsData.requests[sessionId];
    if (!request) return;

    // Load content into each phase panel (except status and everything)
    ['preflight', 'generation', 'qa', 'consensus', 'gransabio'].forEach(phase => {
        const contentEl = document.getElementById(`content-${phase}`);
        if (contentEl) {
            contentEl.textContent = request.content[phase] || '';
            contentEl.scrollTop = contentEl.scrollHeight;
        }

        // Update stats display
        const reqStats = request.stats[phase];
        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = reqStats.chunks;
        if (bytesEl) bytesEl.textContent = reqStats.bytes;
    });
}

function updateRequestInfoBar() {
    const nameEl = document.getElementById('currentRequestName');
    const sessionEl = document.getElementById('currentSessionId');

    // Overview mode
    if (isOverviewMode()) {
        const activeCount = Object.values(requestsData.requests)
            .filter(r => r.status === 'active').length;
        const totalCount = requestsData.orderedRequestIds.length;

        if (nameEl) nameEl.textContent = 'All Sessions';
        if (sessionEl) {
            sessionEl.textContent = `${activeCount} active / ${totalCount} total`;
            sessionEl.title = 'Overview mode - select a tab to view specific session';
        }
        return;
    }

    // Session mode
    const request = requestsData.requests[viewState.selectedRequestId];

    if (request) {
        if (nameEl) nameEl.textContent = request.requestName;
        if (sessionEl) {
            sessionEl.textContent = request.sessionId.slice(0, 12) + '...';
            sessionEl.title = request.sessionId;
        }
    } else {
        if (nameEl) nameEl.textContent = '-';
        if (sessionEl) {
            sessionEl.textContent = '-';
            sessionEl.title = '';
        }
    }
}

// ============================================
// OVERFLOW MENU
// ============================================

function renderOverflowMenu(overflowRequests) {
    const menu = document.getElementById('tabsOverflowMenu');
    if (!menu) return;

    menu.innerHTML = '';

    overflowRequests.forEach(sessionId => {
        const request = requestsData.requests[sessionId];
        const item = document.createElement('div');
        item.className = 'tabs-overflow-item';
        item.title = `${request.requestName}\nSession: ${sessionId}`;

        let statusIcon;
        switch (request.status) {
            case 'completed': statusIcon = '[+]'; break;
            case 'failed': statusIcon = '[x]'; break;
            case 'pending': statusIcon = '[o]'; break;
            default: statusIcon = '[*]';
        }

        item.innerHTML = `
            <span class="overflow-status">${statusIcon}</span>
            <span class="overflow-name">${escapeHtml(request.requestName)}</span>
        `;

        item.onclick = () => {
            selectRequest(sessionId);
            toggleOverflowMenu();
        };

        menu.appendChild(item);
    });
}

function toggleOverflowMenu() {
    const menu = document.getElementById('tabsOverflowMenu');
    if (menu) {
        menu.classList.toggle('hidden');
    }
}

// ============================================
// PROJECT STATUS DASHBOARD
// ============================================

function showDashboard() {
    const dashboard = document.getElementById('statusDashboard');
    if (dashboard) dashboard.classList.remove('hidden');
}

function hideDashboard() {
    const dashboard = document.getElementById('statusDashboard');
    if (dashboard) dashboard.classList.add('hidden');
}

function renderProjectStatus(projectData, filterSessionId = undefined) {
    if (!projectData) return;

    // Cache for re-rendering when switching tabs
    lastProjectData = projectData;

    showDashboard();

    // Determine filter: use provided value, or fall back to current viewState
    if (filterSessionId === undefined) {
        filterSessionId = viewState.selectedRequestId;
    }

    const isOverview = (filterSessionId === null);

    // Update dashboard title based on mode
    const dashboardTitle = document.querySelector('.dashboard-title');
    if (dashboardTitle) {
        if (isOverview) {
            dashboardTitle.textContent = '// All Sessions Overview';
            dashboardTitle.className = 'dashboard-title';
        } else {
            const request = requestsData.requests[filterSessionId];
            const name = request?.requestName || 'Session';
            dashboardTitle.textContent = `// ${name} Status`;
            dashboardTitle.className = 'dashboard-title';
        }
    }

    // Update project ID
    const projectIdEl = document.getElementById('dashboardProjectId');
    if (projectIdEl) {
        projectIdEl.textContent = projectData.project_id || '-';
    }

    // Update project status badge
    const statusBadgeEl = document.getElementById('dashboardProjectStatus');
    if (statusBadgeEl) {
        const status = (projectData.status || 'idle').toLowerCase();
        statusBadgeEl.textContent = status.toUpperCase();
        statusBadgeEl.className = `project-status-badge ${status}`;
    }

    // Render sessions
    const sessionsContainer = document.getElementById('sessionsContainer');
    if (!sessionsContainer) return;

    let sessions = projectData.sessions || [];

    // Filter sessions if not in overview mode
    if (!isOverview && filterSessionId) {
        sessions = sessions.filter(s => s.session_id === filterSessionId);
    }

    if (sessions.length === 0) {
        const message = isOverview
            ? 'No active sessions'
            : 'Session data not available yet';
        sessionsContainer.innerHTML = `<div class="no-sessions">${message}</div>`;
        return;
    }

    sessionsContainer.innerHTML = '';
    sessions.forEach(session => {
        const card = renderSessionCard(session);
        sessionsContainer.appendChild(card);
    });
}

function renderSessionCard(session) {
    const card = document.createElement('div');
    card.className = 'session-card';
    card.id = `session-card-${session.session_id}`;

    const sessionIdShort = (session.session_id || '').slice(0, 12);
    const status = session.status || 'unknown';
    const statusClass = status.toLowerCase().replace(/\s+/g, '_');

    // Determine phase states
    const phases = ['initializing', 'generating', 'qa_evaluation', 'consensus', 'completed'];
    const currentPhaseIndex = phases.indexOf(session.phase || session.status || '');

    card.innerHTML = `
        <div class="session-card-header">
            <div class="session-id-label">SESSION: <span>${sessionIdShort}</span></div>
            <div class="session-status-badge ${statusClass}">${status}</div>
        </div>
        <div class="session-card-body">
            ${renderPhaseTracker(session, phases, currentPhaseIndex)}
            ${renderIterationProgress(session)}
            ${renderGenerationInfo(session)}
            ${renderQASection(session)}
            ${renderConsensusSection(session)}
            ${renderGranSabioSection(session)}
        </div>
    `;

    return card;
}

function renderPhaseTracker(session, phases, currentIndex) {
    const phaseLabels = {
        'initializing': 'INIT',
        'generating': 'GEN',
        'qa_evaluation': 'QA',
        'consensus': 'CONS',
        'completed': 'DONE'
    };

    let html = '<div class="phase-tracker">';
    phases.forEach((phase, index) => {
        let stateClass = 'pending';
        if (index < currentIndex) stateClass = 'completed';
        else if (index === currentIndex) stateClass = 'active';

        html += `<span class="phase-pill ${stateClass}">${phaseLabels[phase] || phase}</span>`;
    });
    html += '</div>';
    return html;
}

function renderIterationProgress(session) {
    const current = session.iteration || 1;
    const max = session.max_iterations || 5;
    const percent = Math.min((current / max) * 100, 100);

    return `
        <div class="progress-row">
            <span class="progress-label">Iteration</span>
            <div class="progress-bar-container">
                <div class="progress-bar-fill iteration" style="width: ${percent}%"></div>
            </div>
            <span class="progress-value">${current}/${max}</span>
        </div>
    `;
}

function renderGenerationInfo(session) {
    const gen = session.generation || {};
    if (!gen.model && !gen.word_count) return '';

    return `
        <div class="info-section">
            <div class="info-section-title">Generation</div>
            <div class="info-grid">
                ${gen.model ? `<div class="info-item"><span class="info-item-label">Model:</span><span class="info-item-value">${gen.model}</span></div>` : ''}
                ${gen.word_count !== undefined ? `<div class="info-item"><span class="info-item-label">Words:</span><span class="info-item-value">${gen.word_count.toLocaleString()}</span></div>` : ''}
                ${gen.content_length !== undefined ? `<div class="info-item"><span class="info-item-label">Chars:</span><span class="info-item-value">${gen.content_length.toLocaleString()}</span></div>` : ''}
            </div>
        </div>
    `;
}

function renderQASection(session) {
    const qa = session.qa || {};
    if (!qa.models && !qa.layers) return '';

    const models = qa.models || [];
    const layers = qa.layers || [];
    const currentModel = qa.current_model || '';
    const currentLayer = qa.current_layer || '';
    const progress = qa.progress || {};

    let html = '<div class="info-section">';
    html += '<div class="info-section-title">QA Evaluation</div>';

    // Models row
    if (models.length > 0) {
        html += '<div style="margin-bottom: 8px;"><span class="info-item-label" style="font-size: 9px;">Models:</span></div>';
        html += '<div class="model-pills">';
        models.forEach(model => {
            const isActive = model === currentModel;
            const isCompleted = models.indexOf(model) < models.indexOf(currentModel);
            let stateClass = isActive ? 'active' : (isCompleted ? 'completed' : '');
            let icon = isActive ? '*' : (isCompleted ? '+' : 'o');
            html += `<span class="model-pill ${stateClass}"><span class="status-icon">${icon}</span>${model}</span>`;
        });
        html += '</div>';
    }

    // Layers row
    if (layers.length > 0) {
        html += '<div style="margin: 8px 0;"><span class="info-item-label" style="font-size: 9px;">Layers:</span></div>';
        html += '<div class="layer-pills">';
        layers.forEach(layer => {
            const isActive = layer === currentLayer;
            const isCompleted = layers.indexOf(layer) < layers.indexOf(currentLayer);
            let stateClass = isActive ? 'active' : (isCompleted ? 'completed' : '');
            let icon = isActive ? '*' : (isCompleted ? '+' : 'o');
            html += `<span class="layer-pill ${stateClass}"><span class="status-icon">${icon}</span>${layer}</span>`;
        });
        html += '</div>';
    }

    // Progress bar
    if (progress.total) {
        const completed = progress.completed || 0;
        const total = progress.total || 1;
        const percent = Math.min((completed / total) * 100, 100);

        html += `
            <div class="progress-row" style="margin-top: 10px;">
                <span class="progress-label">Progress</span>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill qa" style="width: ${percent}%"></div>
                </div>
                <span class="progress-value">${completed}/${total}</span>
            </div>
        `;
    }

    html += '</div>';
    return html;
}

function renderConsensusSection(session) {
    const consensus = session.consensus || {};
    if (consensus.last_score === undefined && consensus.min_required === undefined) return '';

    const score = consensus.last_score || 0;
    const minRequired = consensus.min_required || 8.0;
    const approved = consensus.approved;
    const isAbove = score >= minRequired;

    // Calculate positions (0-10 scale)
    const scorePercent = Math.min((score / 10) * 100, 100);
    const thresholdPercent = (minRequired / 10) * 100;

    let statusText = approved === true ? 'APPROVED' : (approved === false ? 'BELOW THRESHOLD' : 'PENDING');
    if (approved === undefined && score > 0) {
        statusText = isAbove ? 'ABOVE MIN' : 'BELOW MIN';
    }

    return `
        <div class="info-section">
            <div class="info-section-title">Consensus</div>
            <div class="info-grid" style="margin-bottom: 8px;">
                <div class="info-item">
                    <span class="info-item-label">Score:</span>
                    <span class="info-item-value">${score.toFixed(1)}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Min:</span>
                    <span class="info-item-value">${minRequired.toFixed(1)}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Status:</span>
                    <span class="info-item-value" style="color: ${isAbove ? 'var(--green)' : 'var(--amber)'}">${statusText}</span>
                </div>
            </div>
            <div class="consensus-gauge">
                <div class="gauge-bar">
                    <div class="gauge-fill ${isAbove ? 'above' : 'below'}" style="width: ${scorePercent}%"></div>
                    <div class="gauge-threshold" style="left: ${thresholdPercent}%"></div>
                    ${score > 0 ? `<div class="gauge-current" style="left: ${scorePercent}%">${score.toFixed(1)}</div>` : ''}
                </div>
                <div class="gauge-labels">
                    <span>0</span>
                    <span>5</span>
                    <span>10</span>
                </div>
            </div>
        </div>
    `;
}

function renderGranSabioSection(session) {
    const gs = session.gran_sabio || {};
    if (!gs.model && gs.escalation_count === undefined) return '';

    const isActive = gs.active || false;
    const model = gs.model || 'claude-opus-4';
    const escalations = gs.escalation_count || 0;
    const maxEscalations = 15; // Default from config

    return `
        <div class="info-section">
            <div class="info-section-title">Gran Sabio</div>
            <div class="gran-sabio-status">
                <div class="gran-sabio-indicator">
                    <span class="gran-sabio-dot ${isActive ? 'active' : 'standby'}"></span>
                    <span>${isActive ? 'ACTIVE' : 'STANDBY'}</span>
                </div>
                <div class="gran-sabio-model">Model: <span>${model}</span></div>
                <div class="gran-sabio-escalations">Escalations: <span>${escalations}/${maxEscalations}</span></div>
            </div>
        </div>
    `;
}

// ============================================
// URL MANAGEMENT
// ============================================

function getProjectFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get('project');
}

function updateUrl(projectId) {
    const url = new URL(window.location);
    if (projectId) {
        url.searchParams.set('project', projectId);
    } else {
        url.searchParams.delete('project');
    }
    window.history.pushState({}, '', url);
}

// ============================================
// RECENT PROJECTS (ACTIVE CONNECTIONS)
// ============================================

async function loadRecentProjects() {
    const listEl = document.getElementById('recentList');
    const countEl = document.getElementById('recentCount');

    try {
        const response = await fetch('/monitor/active');
        const data = await response.json();

        const projects = data.projects || [];
        const standalone = data.standalone_sessions || [];
        const totalCount = projects.length + standalone.length;

        countEl.textContent = `${projects.length} projects, ${standalone.length} sessions`;

        if (totalCount === 0) {
            listEl.innerHTML = '<div class="empty-message">No active projects or sessions</div>';
            return;
        }

        listEl.innerHTML = '';

        // Render projects first
        projects.forEach(project => {
            const item = document.createElement('div');
            item.className = 'recent-item';

            const statusClass = project.status === 'running' ? 'active' : '';

            item.innerHTML = `
                <div class="recent-info">
                    <div class="recent-name">[PROJECT] ${escapeHtml(project.project_id)}</div>
                    <div class="recent-meta">
                        Sessions: ${project.session_count} |
                        Active: ${project.active_sessions} |
                        Status: ${project.status}
                    </div>
                    <div class="recent-project-id">${project.project_id}</div>
                </div>
                <button class="btn btn-connect btn-small" onclick="connectToProject('${escapeHtml(project.project_id)}')">
                    Connect
                </button>
            `;
            listEl.appendChild(item);
        });

        // Render standalone sessions
        standalone.forEach(session => {
            const item = document.createElement('div');
            item.className = 'recent-item';

            item.innerHTML = `
                <div class="recent-info">
                    <div class="recent-name">[SESSION] ${escapeHtml(session.request_name || 'Unnamed')}</div>
                    <div class="recent-meta">
                        Phase: ${session.phase || '?'} |
                        Status: ${session.status || '?'}
                    </div>
                    <div class="recent-project-id" style="color: var(--amber);">${session.session_id}</div>
                </div>
                <button class="btn btn-connect btn-small" onclick="connectToSession('${session.session_id}')">
                    Connect
                </button>
            `;
            listEl.appendChild(item);
        });

        log(`Loaded ${projects.length} projects, ${standalone.length} standalone sessions`, 'success');

    } catch (error) {
        log(`Failed to load active connections: ${error.message}`, 'error');
        listEl.innerHTML = `<div class="empty-message" style="color: var(--red);">Error: ${escapeHtml(error.message)}</div>`;
    }
}

// ============================================
// CONNECTION (DUAL MODE)
// ============================================

function connectToSession(sessionId) {
    // For standalone sessions, we connect to /stream/{session_id} instead of project stream
    currentConnectionMode = 'session';
    currentSessionId = sessionId;
    document.getElementById('projectIdInput').value = sessionId;
    connect();
}

function connectToProject(projectId) {
    currentConnectionMode = 'project';
    currentSessionId = null;
    document.getElementById('projectIdInput').value = projectId;
    connect();
}

function connect() {
    const inputValue = document.getElementById('projectIdInput').value.trim();

    if (!inputValue) {
        log('Please enter a project ID or session ID', 'error');
        return;
    }

    // Disconnect existing connection
    if (eventSource) {
        disconnect();
    }

    clearAllContent();
    setConnectionStatus('connecting', 'CONNECTING...');

    let streamUrl;

    if (currentConnectionMode === 'session') {
        // Single session mode - simpler stream
        currentProjectId = null;
        currentSessionId = inputValue;
        streamUrl = `${STREAM_BASE}/stream/${inputValue}`;
        log(`Connecting to session stream: ${streamUrl}`, 'info');

        // Hide multi-session UI elements in session mode
        document.getElementById('requestTabsPanel').style.display = 'none';

    } else {
        // Project mode - full multiplexed stream
        currentProjectId = inputValue;
        currentSessionId = null;

        const activePhases = getActivePhases();
        const phasesParam = activePhases.length === HARD_SWITCHABLE_PHASES.length ? 'all' : activePhases.join(',');
        streamUrl = `${STREAM_BASE}/stream/project/${currentProjectId}?phases=${phasesParam}`;

        log(`Connecting to project stream: ${streamUrl}`, 'info');
        updateUrl(inputValue);
    }

    try {
        eventSource = new EventSource(streamUrl);

        eventSource.onopen = function() {
            log('EventSource connection opened', 'success');
            setConnectionStatus('connected', 'ONLINE');
            PHASES.forEach(phase => setBadge(phase, 'active'));
        };

        eventSource.onmessage = function(event) {
            if (currentConnectionMode === 'session') {
                handleSessionMessage(event.data);
            } else {
                handleMessage(event.data);
            }
        };

        eventSource.onerror = function(error) {
            handleConnectionError(error);
        };

    } catch (error) {
        log(`Failed to connect: ${error.message}`, 'error');
        setConnectionStatus('disconnected', 'ERROR');
    }
}

function disconnect() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
        log('Disconnected from stream', 'info');
    }

    currentProjectId = null;
    currentSessionId = null;
    currentConnectionMode = 'project';
    setConnectionStatus('disconnected', 'OFFLINE');
    PHASES.forEach(phase => setBadge(phase, 'idle'));
    updateUrl(null);
}

// ============================================
// MESSAGE HANDLING (PROJECT MODE)
// ============================================

function handleMessage(rawData) {
    // Always append raw data to "everything" panel (unfiltered)
    appendContent('everything', rawData + '\n');

    try {
        const data = JSON.parse(rawData);
        const eventType = data.type;
        const phase = data.phase || 'status';

        // Log event type (except chunks to avoid spam)
        if (eventType !== 'chunk') {
            log(`Event: ${eventType} (phase: ${phase})`, 'event');
        }

        switch (eventType) {
            case 'connected':
                log(`Connected to project ${data.project_id}`, 'success');
                log(`Subscribed phases: ${data.subscribed_phases?.join(', ') || 'all'}`, 'info');
                break;

            case 'status_snapshot':
                handleStatusSnapshot(data);
                break;

            case 'chunk':
                handleChunk(data);
                break;

            case 'status_change':
                handleStatusChange(data);
                break;

            case 'session_end':
                log(`Session ${data.session_id?.slice(0, 8) || '?'} ended`, 'warn');
                appendContent('status', `[SESSION_END] ${JSON.stringify(data, null, 2)}\n`);
                break;

            case 'stream_end':
                log(`Stream ended: ${data.reason}`, 'warn');
                appendContent('status', `[STREAM_END] Reason: ${data.reason}\n`);
                setConnectionStatus('disconnected', 'STREAM ENDED');
                break;

            default:
                // Unknown event type - show in status panel
                log(`Unknown event type: ${eventType}`, 'warn');
                appendContent('status', `[${eventType}] ${JSON.stringify(data, null, 2)}\n`);
        }

    } catch (parseError) {
        // Non-JSON data - show raw
        log(`Parse error: ${parseError.message}`, 'warn');
        appendContent('status', rawData + '\n');
    }
}

function handleChunk(data) {
    // Process and track by request
    const request = processRequestFromEvent(data);

    // Normalize phase name: gran_sabio -> gransabio (match HTML element IDs)
    let phase = data.phase || 'generation';
    if (phase === 'gran_sabio') {
        phase = 'gransabio';
    }
    const content = data.content || '';

    if (!content) return;

    // Store content in request's data structure (for phases we track)
    if (request && phase !== 'status' && request.content.hasOwnProperty(phase)) {
        request.content[phase] = (request.content[phase] || '') + content;
        request.stats[phase].chunks++;
        request.stats[phase].bytes += content.length;
    }

    // In Overview mode: don't display in panels, but show generic activity
    if (isOverviewMode()) {
        if (request) {
            // Show activity indicator (generic receiving badge)
            showReceiving(phase);
        }
        return;
    }

    // Session mode: display only if this is the selected request
    if (request && request.sessionId === viewState.selectedRequestId) {
        appendContent(phase, content);
    } else if (!request) {
        // No session_id in data - still show (legacy behavior)
        appendContent(phase, content);
    }

    // Note: 'everything' panel receives all content via handleMessage
    // before this function is called, so it remains unfiltered
}

function handleStatusSnapshot(data) {
    log('Received status snapshot', 'info');
    const project = data.project || {};
    const summary = project.summary || {};

    // Render visual dashboard
    renderProjectStatus(project);

    // Also output to text panel for debugging
    const statusText = `[STATUS SNAPSHOT]
Project: ${project.project_id || '?'}
Status: ${project.status || '?'}
Sessions: ${summary.total || 0} total, ${summary.active || 0} active, ${summary.completed || 0} completed
`;
    appendContent('status', statusText);

    // Show session details if available
    if (project.sessions && project.sessions.length > 0) {
        project.sessions.forEach(session => {
            appendContent('status', `  Session ${session.session_id?.slice(0, 8) || '?'}: ${session.status} (${session.phase || '?'})\n`);
        });
    }
}

function handleStatusChange(data) {
    // Update request tracking from status change
    const request = processRequestFromEvent(data);
    if (request) {
        // Update request state from status change event
        if (data.iteration !== undefined) {
            request.currentIteration = data.iteration;
        }
        if (data.status) {
            request.status = data.status;
        }
        if (data.score !== undefined) {
            request.finalScore = data.score;
        }

        // Re-render tabs to show updated status
        renderRequestTabs();
    }

    // Project dashboard update (existing logic)
    const project = data.project || {};
    const triggeredBy = data.trigger_session?.slice(0, 8) || '?';

    // Re-render visual dashboard with updated data
    renderProjectStatus(project);

    // Also output to text panel for debugging
    appendContent('status', `[STATUS_CHANGE] Triggered by session ${triggeredBy}\n`);

    if (project.summary) {
        appendContent('status', `  Active: ${project.summary.active || 0}, Completed: ${project.summary.completed || 0}\n`);
    }
}

// ============================================
// MESSAGE HANDLING (SESSION MODE)
// ============================================

function handleSessionMessage(rawData) {
    // Always append to "everything" panel
    appendContent('everything', rawData + '\n');

    try {
        const data = JSON.parse(rawData);

        // Session stream format is different from project stream
        // It sends ProgressUpdate objects

        if (data.status) {
            const statusStr = data.status.value || data.status;
            appendContent('status', `[STATUS] ${statusStr}\n`);
        }

        if (data.generated_content) {
            // Full content replacement
            const genContent = document.getElementById('content-generation');
            if (genContent) {
                genContent.textContent = data.generated_content;
                genContent.scrollTop = genContent.scrollHeight;
            }
            showReceiving('generation');
        }

        if (data.qa_feedback) {
            const qaContent = document.getElementById('content-qa');
            if (qaContent) {
                qaContent.textContent = JSON.stringify(data.qa_feedback, null, 2);
            }
            showReceiving('qa');
        }

        if (data.verbose_log && Array.isArray(data.verbose_log)) {
            data.verbose_log.forEach(logEntry => {
                log(logEntry, 'event');
            });
        }

        // Update status panel with iteration info
        if (data.current_iteration !== undefined) {
            appendContent('status', `Iteration: ${data.current_iteration}/${data.max_iterations || '?'}\n`);
        }

        // Check for completion
        const statusStr = data.status?.value || data.status;
        if (statusStr === 'completed' || statusStr === 'failed' || statusStr === 'cancelled') {
            log(`Session ended: ${statusStr}`, 'warn');
            setConnectionStatus('disconnected', statusStr.toUpperCase());
        }

    } catch (parseError) {
        log(`Parse error: ${parseError.message}`, 'warn');
        appendContent('status', rawData + '\n');
    }
}

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    log('BioAI Stream Monitor initialized', 'success');
    log(`Stream endpoint: ${STREAM_BASE}/stream/project/{id}`, 'info');

    // Initialize toggle switch listeners
    initToggleListeners();
    log('Phase toggle switches ready', 'info');

    // Load active connections
    loadRecentProjects();

    // Check URL for project parameter
    const urlProject = getProjectFromUrl();
    if (urlProject) {
        log(`Found project in URL: ${urlProject}`, 'info');
        document.getElementById('projectIdInput').value = urlProject;
        currentConnectionMode = 'project';
        // Auto-connect after a short delay
        setTimeout(() => connect(), 500);
    }

    // Allow Enter key to connect
    document.getElementById('projectIdInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            // Determine mode based on input format (UUIDs are typically sessions)
            const value = e.target.value.trim();
            if (value.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) {
                // Looks like a UUID - treat as session
                currentConnectionMode = 'session';
            } else {
                // Treat as project ID
                currentConnectionMode = 'project';
            }
            connect();
        }
    });

    // Close overflow menu when clicking outside
    document.addEventListener('click', (e) => {
        const menu = document.getElementById('tabsOverflowMenu');
        const overflowBtn = document.querySelector('.tabs-overflow-btn');
        if (menu && !menu.contains(e.target) && e.target !== overflowBtn) {
            menu.classList.add('hidden');
        }
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
    }
});
