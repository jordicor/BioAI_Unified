const defaultProjectId = (document.body.dataset.projectId || "").trim();

/**
 * Escapes HTML special characters to prevent XSS attacks.
 * @param {string} str - The string to escape
 * @returns {string} - The escaped string safe for innerHTML
 */
function escapeHtml(str) {
    if (str === null || str === undefined) {
        return '';
    }
    const text = String(str);
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

const state = {
    sessions: [],
    filteredSessions: [],
    selectedSession: null,
    eventOffset: 0,
    eventLimit: 50,
    projectId: defaultProjectId,
};

let projectFilterDebounce = null;

async function fetchSessions() {
    try {
        const params = new URLSearchParams({ limit: state.eventLimit });
        if (state.projectId) {
            params.append("project_id", state.projectId);
        }
        const response = await fetch(`/debugger/sessions?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`Failed to load sessions (${response.status})`);
        }
        const data = await response.json();
        state.sessions = data.sessions || [];
        if (state.projectId !== (data.project_id || "")) {
            state.projectId = (data.project_id || "").trim();
        }
        if (!state.sessions.some((session) => session.session_id === state.selectedSession)) {
            state.selectedSession = null;
        }
        applySessionFilter();
        const projectInput = document.getElementById("project-filter");
        if (projectInput && projectInput !== document.activeElement) {
            projectInput.value = state.projectId;
        }
    } catch (error) {
        console.error(error);
    }
}

function applySessionFilter() {
    const filterValue = document.getElementById("session-filter").value.trim().toLowerCase();
    if (!filterValue) {
        state.filteredSessions = state.sessions.slice();
    } else {
        state.filteredSessions = state.sessions.filter((session) =>
            session.session_id.toLowerCase().includes(filterValue) ||
            (session.request_name && session.request_name.toLowerCase().includes(filterValue))
        );
    }
    renderSessions();
}

function renderSessions() {
    const list = document.getElementById("session-list");
    list.innerHTML = "";
    state.filteredSessions.forEach((session) => {
        const item = document.createElement("li");
        const projectLabel = session.project_id ? ` | project ${session.project_id}` : "";
        const requestNameLabel = session.request_name ? ` | ${session.request_name}` : "";
        item.textContent = `${session.session_id} | ${session.status || "unknown"}${projectLabel}${requestNameLabel}`;
        item.dataset.sessionId = session.session_id;
        if (state.selectedSession === session.session_id) {
            item.classList.add("active");
        }
        item.addEventListener("click", () => selectSession(session.session_id));
        list.appendChild(item);
    });
}

function formatJSON(payload) {
    if (payload === null || payload === undefined) {
        return "";
    }
    if (typeof payload === "string") {
        return payload;
    }
    try {
        return JSON.stringify(payload, null, 2);
    } catch (error) {
        return String(payload);
    }
}

/**
 * Renders the Evidence Grounding panel with claim verification results.
 * @param {Object|null} grounding - The evidence_grounding object from final result
 */
function renderGroundingPanel(grounding) {
    const section = document.getElementById("grounding-section");
    const summaryEl = document.getElementById("grounding-summary");
    const claimsEl = document.getElementById("grounding-claims");

    // Hide section if no grounding data
    if (!grounding || !grounding.enabled) {
        section.classList.add("hidden");
        summaryEl.innerHTML = "";
        claimsEl.innerHTML = "";
        return;
    }

    section.classList.remove("hidden");

    // Determine status class
    const statusClass = grounding.passed ? "grounding-passed" : "grounding-failed";
    const statusText = grounding.passed ? "PASSED" : "FAILED";
    const statusIcon = grounding.passed ? "OK" : "X";

    // Build summary HTML
    const summaryHtml = `
        <div class="grounding-status ${statusClass}">
            <span class="status-icon">${statusIcon}</span>
            <span class="status-text">${statusText}</span>
        </div>
        <div class="grounding-metrics">
            <div class="metric">
                <span class="metric-label">Model</span>
                <span class="metric-value">${escapeHtml(grounding.model_used || "N/A")}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Claims Verified</span>
                <span class="metric-value">${grounding.claims_verified || 0} / ${grounding.total_claims_extracted || 0}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Flagged Claims</span>
                <span class="metric-value ${grounding.flagged_claims > 0 ? 'flagged' : ''}">${grounding.flagged_claims || 0}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Max Budget Gap</span>
                <span class="metric-value">${(grounding.max_budget_gap || 0).toFixed(3)} bits</span>
            </div>
            <div class="metric">
                <span class="metric-label">Verification Time</span>
                <span class="metric-value">${(grounding.verification_time_ms || 0).toFixed(0)} ms</span>
            </div>
            ${grounding.triggered_action ? `
            <div class="metric">
                <span class="metric-label">Triggered Action</span>
                <span class="metric-value action-${grounding.triggered_action}">${grounding.triggered_action}</span>
            </div>
            ` : ''}
        </div>
    `;
    summaryEl.innerHTML = summaryHtml;

    // Build claims table if there are claims
    const claims = grounding.claims || [];
    if (claims.length === 0) {
        claimsEl.innerHTML = '<p class="no-claims">No claims were verified.</p>';
        return;
    }

    let claimsHtml = `
        <table class="grounding-claims-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Claim</th>
                    <th>P(YES|ctx)</th>
                    <th>P(YES|no-ctx)</th>
                    <th>Budget Gap</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
    `;

    claims.forEach((claim) => {
        const rowClass = claim.flagged ? "claim-flagged" : "claim-ok";
        const statusBadge = claim.flagged
            ? '<span class="badge badge-flagged">FLAGGED</span>'
            : '<span class="badge badge-ok">OK</span>';

        // Truncate long claims for display
        const claimText = claim.claim || "";
        const displayClaim = claimText.length > 80
            ? escapeHtml(claimText.substring(0, 80)) + "..."
            : escapeHtml(claimText);

        claimsHtml += `
            <tr class="${rowClass}" title="${escapeHtml(claimText)}">
                <td>${claim.idx}</td>
                <td class="claim-text">${displayClaim}</td>
                <td>${(claim.posterior_yes || 0).toFixed(3)}</td>
                <td>${(claim.prior_yes || 0).toFixed(3)}</td>
                <td class="${claim.budget_gap > 0.5 ? 'gap-high' : 'gap-ok'}">${(claim.budget_gap || 0).toFixed(3)}</td>
                <td>${statusBadge}</td>
            </tr>
        `;
    });

    claimsHtml += `
            </tbody>
        </table>
    `;
    claimsEl.innerHTML = claimsHtml;
}

async function selectSession(sessionId) {
    state.selectedSession = sessionId;
    renderSessions();
    await loadSessionDetails(sessionId);
    await loadEvents(true);
}

async function loadSessionDetails(sessionId) {
    try {
        const response = await fetch(`/debugger/sessions/${sessionId}`);
        if (!response.ok) {
            throw new Error(`Failed to load session details (${response.status})`);
        }
        const details = await response.json();
        document.getElementById("session-title").textContent = `Session ${details.session_id}`;
        const safeStatus = escapeHtml(details.status || "unknown");
        const safeCreated = escapeHtml(details.created_at || "n/a");
        const safeUpdated = escapeHtml(details.updated_at || "n/a");
        const safeProject = escapeHtml(details.project_id || "—");
        const safeRequestName = escapeHtml(details.request_name || "—");
        document.getElementById("session-summary").innerHTML = `
            <span><strong>Status:</strong> ${safeStatus}</span>
            <span><strong>Created:</strong> ${safeCreated}</span>
            <span><strong>Updated:</strong> ${safeUpdated}</span>
            <span><strong>Project:</strong> ${safeProject}</span>
            <span><strong>Request Name:</strong> ${safeRequestName}</span>
        `;
        document.getElementById("request-data").value = formatJSON(details.request);
        document.getElementById("preflight-data").value = formatJSON(details.preflight);
        document.getElementById("final-data").value = formatJSON(details.final);
        document.getElementById("usage-data").value = formatJSON(details.usage);

        // Render Evidence Grounding panel if data is available
        const finalResult = details.final || {};
        const groundingData = finalResult.evidence_grounding || null;
        renderGroundingPanel(groundingData);
    } catch (error) {
        console.error(error);
    }
}

async function loadEvents(reset = false) {
    if (!state.selectedSession) {
        return;
    }
    if (reset) {
        state.eventOffset = 0;
        document.getElementById("events-list").innerHTML = "";
    }
    try {
        const response = await fetch(
            `/debugger/sessions/${state.selectedSession}/events?offset=${state.eventOffset}&limit=${state.eventLimit}`
        );
        if (!response.ok) {
            throw new Error(`Failed to load events (${response.status})`);
        }
        const data = await response.json();
        const events = data.events || [];
        renderEvents(events, reset);
        if (events.length === state.eventLimit) {
            state.eventOffset += state.eventLimit;
            document.getElementById("load-more-events").classList.remove("hidden");
        } else {
            document.getElementById("load-more-events").classList.add("hidden");
        }
    } catch (error) {
        console.error(error);
    }
}

function renderEvents(events, reset) {
    const container = document.getElementById("events-list");
    if (reset) {
        container.innerHTML = "";
    }
    events.forEach((event) => {
        const wrapper = document.createElement("div");
        wrapper.className = "event-entry";

        const header = document.createElement("header");
        const title = document.createElement("h4");
        title.textContent = `${event.event_order}. ${event.event_type}`;
        const timestamp = document.createElement("span");
        timestamp.textContent = event.created_at || "";
        header.appendChild(title);
        header.appendChild(timestamp);

        const body = document.createElement("pre");
        body.textContent = formatJSON(event.payload);

        wrapper.appendChild(header);
        wrapper.appendChild(body);
        container.appendChild(wrapper);
    });
}

function setupEventHandlers() {
    document.getElementById("refresh-sessions").addEventListener("click", fetchSessions);
    document.getElementById("session-filter").addEventListener("input", applySessionFilter);
    const projectFilterInput = document.getElementById("project-filter");
    if (projectFilterInput) {
        projectFilterInput.value = state.projectId;
        projectFilterInput.addEventListener("input", (event) => {
            if (projectFilterDebounce) {
                clearTimeout(projectFilterDebounce);
            }
            projectFilterDebounce = setTimeout(() => {
                state.projectId = event.target.value.trim();
                state.selectedSession = null;
                fetchSessions();
            }, 300);
        });
    }
    document.getElementById("load-more-events").addEventListener("click", () => loadEvents(false));
}

async function boot() {
    setupEventHandlers();
    await fetchSessions();
}

document.addEventListener("DOMContentLoaded", boot);
