// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let sseSource = null;
    let logs = []; // Holds the currently loaded/streamed logs (client-side)
    let filteredLogs = [];
    let clusterSummaries = null; // Holds results from /api/analyze
    let errorProfiles = null; // Holds error profiles from /api/analyze
    let currentPage = 1;
    const pageSize = 20; // Number of logs per page in explorer

    // --- DOM Elements ---
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    const dataSourceRadios = document.querySelectorAll('input[name="dataSource"]');
    const streamSettings = document.getElementById('stream-settings');
    const uploadSettings = document.getElementById('upload-settings');
    const sampleSettings = document.getElementById('sample-settings');
    const sseUrlInput = document.getElementById('sse-url');
    const sseStatus = document.getElementById('sse-status');
    const sseError = document.getElementById('sse-error');
    const sseConnectButton = document.getElementById('sse-connect-button');
    const logCountStreamSpan = document.getElementById('log-count-stream');
    const maxStreamLogsInput = document.getElementById('max-stream-logs');
    const logFileInput = document.getElementById('log-file-input');
    const uploadStatus = document.getElementById('upload-status');
    const loadSampleButton = document.getElementById('load-sample-button');
    const sampleStatus = document.getElementById('sample-status');
    const nClustersSlider = document.getElementById('n-clusters');
    const nClustersValue = document.getElementById('n-clusters-value');
    const clusteringInfo = document.getElementById('clustering-info');
    const llmProviderRadios = document.querySelectorAll('input[name="llmProvider"]');
    const ollamaSettings = document.getElementById('ollama-settings');
    const openrouterSettings = document.getElementById('openrouter-settings');
    const ollamaUrlInput = document.getElementById('ollama-url');
    const ollamaModelInput = document.getElementById('ollama-model');
    const openrouterKeyInput = document.getElementById('openrouter-key');
    const openrouterModelSelect = document.getElementById('openrouter-model');
    const chartThemeSelect = document.getElementById('chart-theme');

    const tabs = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');
    const nestedTabs = document.querySelectorAll('.tab-link-nested');
    const nestedTabContents = document.querySelectorAll('.tab-content-nested');

    // Dashboard Metrics
    const metricTotalLogs = document.getElementById('metric-total-logs');
    const metricErrorCount = document.getElementById('metric-error-count');
    const metricErrorRate = document.getElementById('metric-error-rate');
    const metricWarningCount = document.getElementById('metric-warning-count');
    const metricWarningRate = document.getElementById('metric-warning-rate');
    const metricTimeSpan = document.getElementById('metric-time-span');
    const metricUniqueModules = document.getElementById('metric-unique-modules');
    const metricAvgLatency = document.getElementById('metric-avg-latency');

    // Dashboard Charts
    const chartLevelPie = document.getElementById('chart-level-pie');
    const chartTopErrorModules = document.getElementById('chart-top-error-modules');
    const chartLogTimeline = document.getElementById('chart-log-timeline');

    // Log Explorer
    const filterLevelSelect = document.getElementById('filter-level');
    const filterModuleSelect = document.getElementById('filter-module');
    const filterKeywordInput = document.getElementById('filter-keyword');
    const filterStatusSelect = document.getElementById('filter-status');
    const filterEnvSelect = document.getElementById('filter-env');
    const applyFiltersButton = document.getElementById('apply-filters-button');
    const explorerLogCount = document.getElementById('explorer-log-count');
    const logDisplayArea = document.getElementById('log-display-area');
    const pagePrevButton = document.getElementById('page-prev');
    const pageNextButton = document.getElementById('page-next');
    const pageCurrentInput = document.getElementById('page-current');
    const pageTotalSpan = document.getElementById('page-total');

    // AI Analysis
    const runAnalysisButton = document.getElementById('run-analysis-button');
    const analysisStatus = document.getElementById('analysis-status');
    const analysisResultsDiv = document.getElementById('analysis-results');
    const clusterSelector = document.getElementById('cluster-selector');
    const clusterDetailsDiv = document.getElementById('cluster-details');
    const analyzeClusterSummaryButton = document.getElementById('analyze-cluster-summary-button');
    const clusterAnalysisOutput = document.getElementById('cluster-analysis-output');
    const chartClusterDist = document.getElementById('chart-cluster-distribution');
    const chartClusterErrRate = document.getElementById('chart-cluster-error-rate');
    const runHolisticButton = document.getElementById('run-holistic-analysis-button');
    const holisticAnalysisOutput = document.getElementById('holistic-analysis-output');

    // Advanced Viz
    const vizTypeSelector = document.getElementById('viz-type-selector');
    const advancedVizArea = document.getElementById('advanced-viz-area');


    // --- Initialization ---
    updateSidebarView(); // Show settings for default data source
    setupEventListeners();
    // Initial render/clear
    clearLogsAndAnalyses();


    // --- Event Listeners ---
    function setupEventListeners() {
        dataSourceRadios.forEach(radio => radio.addEventListener('change', handleDataSourceChange));
        sseConnectButton.addEventListener('click', handleSseConnectToggle);
        loadSampleButton.addEventListener('click', loadSampleLogs);
        logFileInput.addEventListener('change', handleFileUpload);
        nClustersSlider.addEventListener('input', () => {
            nClustersValue.textContent = nClustersSlider.value;
        });
        llmProviderRadios.forEach(radio => radio.addEventListener('change', updateLlmProviderView));
        tabs.forEach(tab => tab.addEventListener('click', () => switchTab(tab, tabs, tabContents)));
        nestedTabs.forEach(tab => tab.addEventListener('click', () => switchTab(tab, nestedTabs, nestedTabContents)));
        applyFiltersButton.addEventListener('click', applyFiltersAndRenderExplorer);
        pagePrevButton.addEventListener('click', () => changeExplorerPage(-1));
        pageNextButton.addEventListener('click', () => changeExplorerPage(1));
        pageCurrentInput.addEventListener('change', () => {
            const targetPage = parseInt(pageCurrentInput.value, 10);
            if (!isNaN(targetPage) && targetPage >= 1) {
                currentPage = targetPage; // Validation happens in renderExplorer
                renderLogExplorer();
            } else {
                pageCurrentInput.value = currentPage; // Reset if invalid
            }
        });

        runAnalysisButton.addEventListener('click', runClusteringAnalysis);
        clusterSelector.addEventListener('change', displaySelectedClusterDetails);
        analyzeClusterSummaryButton.addEventListener('click', analyzeSelectedClusterSummary);
        runHolisticButton.addEventListener('click', runHolisticAnalysis);

        // Explain button (using event delegation on the display area)
        logDisplayArea.addEventListener('click', handleExplainButtonClick);
    }

    // --- Data Source Handling ---
    function handleDataSourceChange() {
        const selectedSource = document.querySelector('input[name="dataSource"]:checked').value;
        console.log("Data source changed to:", selectedSource);
        stopSseConnection(); // Stop connection if switching away from stream
        clearLogsAndAnalyses();
        updateSidebarView(selectedSource);

        // Reset UI elements associated with previous source
        uploadStatus.textContent = '';
        sampleStatus.textContent = '';
        logFileInput.value = ''; // Clear file input

        // Load sample logs immediately if selected
        if (selectedSource === 'sample') {
            loadSampleLogs();
        }
    }

    function updateSidebarView(source = null) {
        const selectedSource = source || document.querySelector('input[name="dataSource"]:checked').value;
        streamSettings.style.display = selectedSource === 'stream' ? 'block' : 'none';
        uploadSettings.style.display = selectedSource === 'upload' ? 'block' : 'none';
        sampleSettings.style.display = selectedSource === 'sample' ? 'block' : 'none';
    }

    // --- SSE Handling ---
    function handleSseConnectToggle() {
        if (sseSource) {
            stopSseConnection();
        } else {
            startSseConnection();
        }
    }

    function startSseConnection() {
        const url = sseUrlInput.value.trim();
        if (!url) {
            setSseStatus('error', 'SSE URL cannot be empty.');
            return;
        }

        clearLogsAndAnalyses(); // Clear previous logs on new connection
        setSseStatus('connecting', 'Connecting...');
        sseConnectButton.textContent = 'Disconnect';
        sseConnectButton.disabled = true; // Disable while connecting

        try {
             sseSource = new EventSource(url);

             sseSource.onopen = () => {
                 console.log("SSE Connection Opened");
                 setSseStatus('connected', 'Connected');
                 sseConnectButton.disabled = false;
             };

             sseSource.onmessage = (event) => {
                // console.log("SSE Message:", event.data);
                 try {
                     const logEntry = JSON.parse(event.data);
                     // Add a unique ID for the explain button
                     logEntry._id = Date.now() + Math.random();
                     addLogEntry(logEntry);
                 } catch (e) {
                     console.error("Failed to parse SSE data:", e, event.data);
                     // Add as raw parse error?
                     const errorEntry = {
                         _id: Date.now() + Math.random(),
                         timestamp: new Date().toISOString().split('.')[0],
                         level: "PARSE_ERROR",
                         module: "sse_client",
                         message: "Failed to parse incoming SSE message",
                         raw: event.data.substring(0, 200), // Show snippet
                         status_code: "N/A",
                         latency: 0,
                         env: "client_error",
                         datetime: new Date().toISOString()
                     };
                     addLogEntry(errorEntry);
                 }
             };

             sseSource.onerror = (event) => {
                 console.error("SSE Error:", event);
                 let errorMsg = 'Connection failed. Check URL & server status.';
                 if (event.target && event.target.readyState === EventSource.CLOSED) {
                    errorMsg = 'Connection closed by server or network issue.';
                 }
                  // Check if it's a network error before closing immediately
                 if (event.eventPhase === EventSource.CLOSED) {
                     stopSseConnection(errorMsg); // Pass error message
                 } else {
                      setSseStatus('error', errorMsg); // Keep trying if not definitively closed
                      // Could implement exponential backoff here if needed
                 }
                 sseConnectButton.disabled = false; // Re-enable button on error
             };
        } catch(e) {
            console.error("Failed to create EventSource:", e);
            setSseStatus('error', `Failed to create EventSource: ${e.message}`);
            sseConnectButton.textContent = 'Connect';
            sseConnectButton.disabled = false;
        }
    }

    function stopSseConnection(errorMessage = null) {
        if (sseSource) {
            sseSource.close();
            sseSource = null;
            console.log("SSE Connection Closed");
        }
        setSseStatus(errorMessage ? 'error' : 'disconnected', errorMessage || 'Disconnected');
        sseConnectButton.textContent = 'Connect';
        sseConnectButton.disabled = false;
    }

    function setSseStatus(status, message = '') {
        sseStatus.className = `status-indicator status-${status}`;
        sseStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        if (status === 'error' && message) {
            sseError.textContent = `Error: ${message}`;
            sseError.style.display = 'block';
        } else {
            sseError.textContent = '';
            sseError.style.display = 'none';
        }
    }

    // --- Log Loading & Handling ---
     function addLogEntry(logEntry) {
        logs.push(logEntry);

        // Enforce max logs limit
        const maxLogs = parseInt(maxStreamLogsInput.value, 10) || 5000;
        if (logs.length > maxLogs) {
            logs = logs.slice(logs.length - maxLogs);
        }

        // Update UI elements (debounced or throttled for performance)
        // For simplicity, update directly here, but could be optimized
        updateMetricsAndCharts(); // Update metrics/charts with new data
        applyFiltersAndRenderExplorer(); // Re-filter and render explorer if visible
        updateDataSourceCounts();
     }

     function processLoadedLogs(loadedLogArray, sourceName) {
         logs = loadedLogArray.map((log, index) => ({ ...log, _id: `${sourceName}_${index}` })); // Add unique ID
         if (!logs || logs.length === 0) {
             showStatusMessage(`No valid logs found in ${sourceName}.`, 'error', sourceName);
             clearLogsAndAnalyses();
             return;
         }
         showStatusMessage(`Loaded ${logs.length} logs from ${sourceName}.`, 'success', sourceName);
         applyFiltersAndRenderExplorer(); // Initial filter and render
         updateMetricsAndCharts();
         populateFilterDropdowns();
         updateDataSourceCounts();
         enableAnalysisButton();
     }

     function clearLogsAndAnalyses() {
         logs = [];
         filteredLogs = [];
         clusterSummaries = null;
         errorProfiles = null;
         currentPage = 1;
         logDisplayArea.innerHTML = '<p>No logs loaded.</p>';
         analysisResultsDiv.style.display = 'none';
         analysisStatus.textContent = '';
         clusterDetailsDiv.innerHTML = '';
         clusterSelector.innerHTML = '<option>-- Run Analysis --</option>';
         analyzeClusterSummaryButton.style.display = 'none';
         clusterAnalysisOutput.innerHTML = '';
         holisticAnalysisOutput.innerHTML = '';
         advancedVizArea.innerHTML = '';
         vizTypeSelector.innerHTML = '<option value="">-- Select Visualization --</option>';
         updateMetricsAndCharts(); // Reset metrics/charts
         updateDataSourceCounts();
         disableAnalysisButton();
         populateFilterDropdowns(); // Clear or reset filters
     }

     function updateDataSourceCounts() {
         const currentSource = document.querySelector('input[name="dataSource"]:checked').value;
         if (currentSource === 'stream') {
             logCountStreamSpan.textContent = logs.length;
         }
         // Could update counts for upload/sample if needed
     }

    function showStatusMessage(message, type = 'info', source = null) {
         let statusElement = null;
         if (source === 'upload') statusElement = uploadStatus;
         else if (source === 'sample') statusElement = sampleStatus;
         else if (source === 'analysis') statusElement = analysisStatus;

         if (statusElement) {
             statusElement.textContent = message;
             statusElement.className = `status-message ${type}`; // Use CSS classes for styling
         }
         console.log(`Status [${type}]: ${message}`);
     }


     // --- Sample Log Loading ---
     async function loadSampleLogs() {
         clearLogsAndAnalyses();
         showStatusMessage('Loading sample logs...', 'loading', 'sample');
         try {
             const response = await fetch('/api/sample-logs');
             if (!response.ok) {
                 const errorData = await response.json();
                 throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
             }
             const data = await response.json();
             processLoadedLogs(data.logs || [], 'Sample Logs');
         } catch (error) {
             console.error('Error loading sample logs:', error);
             showStatusMessage(`Error loading sample logs: ${error.message}`, 'error', 'sample');
             clearLogsAndAnalyses(); // Clear on error
         }
     }

     // --- File Upload Handling ---
     async function handleFileUpload(event) {
         const file = event.target.files[0];
         if (!file) return;

         clearLogsAndAnalyses();
         showStatusMessage(`Uploading ${file.name}...`, 'loading', 'upload');

         const formData = new FormData();
         formData.append('logFile', file);

         try {
             const response = await fetch('/api/upload', {
                 method: 'POST',
                 body: formData,
             });

             const data = await response.json(); // Try to parse JSON regardless of status

             if (!response.ok) {
                  throw new Error(data.error || `HTTP error! status: ${response.status}`);
             }

             processLoadedLogs(data.logs || [], data.filename || 'Uploaded File');

         } catch (error) {
             console.error('Error uploading file:', error);
             showStatusMessage(`Error uploading file: ${error.message}`, 'error', 'upload');
             clearLogsAndAnalyses(); // Clear on error
         }
     }

    // --- Log Explorer Filtering & Rendering ---
    function populateFilterDropdowns() {
        const levels = [...new Set(logs.map(log => log.level || 'UNKNOWN'))].sort();
        const modules = [...new Set(logs.map(log => log.module || 'unknown'))].sort();
        const statuses = [...new Set(logs.map(log => String(log.status_code || 'N/A')))].sort();
        const envs = [...new Set(logs.map(log => log.env || 'unknown'))].sort();

        populateSelect(filterLevelSelect, levels, "Level");
        populateSelect(filterModuleSelect, modules, "Module");
        populateSelect(filterStatusSelect, statuses.filter(s => s !== 'N/A'), "Status"); // Exclude N/A maybe?
        populateSelect(filterEnvSelect, envs.filter(e => e !== 'unknown'), "Env"); // Exclude unknown
    }

    function populateSelect(selectElement, options, typeName) {
        selectElement.innerHTML = `<option value="All">All ${typeName}s</option>`; // Reset
        options.forEach(option => {
            if (option) { // Avoid adding null/empty options
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                selectElement.appendChild(opt);
            }
        });
    }

    function applyFiltersAndRenderExplorer() {
        const level = filterLevelSelect.value;
        const module = filterModuleSelect.value;
        const keyword = filterKeywordInput.value.toLowerCase();
        const status = filterStatusSelect.value;
        const env = filterEnvSelect.value;

        filteredLogs = logs.filter(log => {
            const levelMatch = level === 'All' || log.level === level;
            const moduleMatch = module === 'All' || log.module === module;
            const statusMatch = status === 'All' || String(log.status_code) === status;
            const envMatch = env === 'All' || log.env === env;
            const keywordMatch = keyword === '' || (log.raw && log.raw.toLowerCase().includes(keyword));

            return levelMatch && moduleMatch && statusMatch && envMatch && keywordMatch;
        });

        currentPage = 1; // Reset to first page on filter change
        pageCurrentInput.value = 1;
        renderLogExplorer();
    }

    function renderLogExplorer() {
        const totalLogs = filteredLogs.length;
        const totalPages = Math.max(1, Math.ceil(totalLogs / pageSize));

        // Validate current page
        if (currentPage > totalPages) currentPage = totalPages;
        if (currentPage < 1) currentPage = 1;
        pageCurrentInput.value = currentPage;

        const startIndex = (currentPage - 1) * pageSize;
        const endIndex = startIndex + pageSize;
        const paginatedLogs = filteredLogs.slice(startIndex, endIndex);

        logDisplayArea.innerHTML = ''; // Clear previous logs

        if (paginatedLogs.length === 0) {
            logDisplayArea.innerHTML = '<p>No logs match the current filters.</p>';
        } else {
            paginatedLogs.forEach((log) => {
                const logCard = document.createElement('div');
                logCard.className = `log-entry-card ${getLogLevelClass(log.level)}`;

                const logText = document.createElement('div');
                logText.className = 'log-text';
                logText.innerHTML = `
                    <span class="timestamp">[${log.timestamp || 'No Timestamp'}]</span>
                    <span class="level-${log.level || 'INFO'}">[${log.level || 'INFO'}]</span>
                    <span class="module">[${log.module || '?'}]</span>
                    <span>${log.message || log.raw || 'No Message'}</span>
                `;

                const explainButton = document.createElement('button');
                explainButton.className = 'explain-button';
                explainButton.textContent = 'Explain';
                explainButton.dataset.logId = log._id; // Use the unique ID

                const explanationBox = document.createElement('div');
                explanationBox.className = 'explanation-box';
                explanationBox.id = `explanation-${log._id}`;
                explanationBox.style.display = 'none'; // Initially hidden

                logCard.appendChild(logText);
                logCard.appendChild(explainButton);

                logDisplayArea.appendChild(logCard);
                logDisplayArea.appendChild(explanationBox); // Add explanation box after the card
            });
        }

        // Update pagination controls
        explorerLogCount.textContent = totalLogs;
        pageTotalSpan.textContent = totalPages;
        pagePrevButton.disabled = currentPage === 1;
        pageNextButton.disabled = currentPage === totalPages;
    }

    function changeExplorerPage(delta) {
        const totalPages = Math.max(1, Math.ceil(filteredLogs.length / pageSize));
        const newPage = currentPage + delta;
        if (newPage >= 1 && newPage <= totalPages) {
            currentPage = newPage;
            pageCurrentInput.value = currentPage;
            renderLogExplorer();
        }
    }

    function getLogLevelClass(level) {
        level = String(level).toUpperCase();
        if (level === 'ERROR') return 'error-card';
        if (level === 'WARNING') return 'warning-card';
        if (level === 'PARSE_ERROR') return 'error-card'; // Treat parse error as error visually
        return 'info-card'; // Default
    }


    // --- Explain Log with AI ---
    async function handleExplainButtonClick(event) {
        if (event.target.classList.contains('explain-button')) {
            const button = event.target;
            const logId = button.dataset.logId;
            const explanationBox = document.getElementById(`explanation-${logId}`);
            const logEntry = logs.find(log => log._id == logId); // Find the log by ID

            if (!logEntry) {
                console.error("Could not find log entry for ID:", logId);
                return;
            }

            // Toggle visibility or fetch explanation
            if (explanationBox.style.display === 'block') {
                explanationBox.style.display = 'none';
                button.textContent = 'Explain';
            } else {
                button.textContent = 'Loading...';
                button.disabled = true;
                explanationBox.innerHTML = '<i>🧠 Thinking...</i>';
                explanationBox.style.display = 'block';

                try {
                    const llmConfig = getLlmConfig();
                    const response = await fetch('/api/llm/explain', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            log_raw: logEntry.raw,
                            use_ollama: llmConfig.useOllama,
                            model: llmConfig.model,
                            api_key: llmConfig.apiKey,
                            ollama_url: llmConfig.ollamaUrl
                        })
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.error || `HTTP ${response.status}`);
                    }
                     // Basic Markdown rendering (replace newline with <br>, bold, etc.)
                     // A proper Markdown library (like Showdown.js or Marked.js) would be better
                    let formattedExplanation = data.explanation.replace(/\n/g, '<br>');
                    formattedExplanation = formattedExplanation.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
                    explanationBox.innerHTML = formattedExplanation;

                } catch (error) {
                    console.error("Error explaining log:", error);
                    explanationBox.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
                } finally {
                     button.textContent = 'Hide Explanation';
                     button.disabled = false;
                }
            }
        }
    }


    // --- Dashboard Metrics & Charts ---
    function updateMetricsAndCharts() {
        // 1. Calculate Metrics
        const total = logs.length;
        let errorCount = 0;
        let warningCount = 0;
        let uniqueModules = new Set();
        let totalLatency = 0;
        let latencyCount = 0;
        let minTime = null;
        let maxTime = null;

        logs.forEach(log => {
            if (log.level === 'ERROR') errorCount++;
            if (log.level === 'WARNING') warningCount++;
            if (log.module) uniqueModules.add(log.module);
            const latency = parseFloat(log.latency);
            if (!isNaN(latency) && latency > 0) {
                totalLatency += latency;
                latencyCount++;
            }
            // Attempt to parse datetime for time span
             try {
                 if (log.datetime) {
                    const dt = new Date(log.datetime);
                     if (!isNaN(dt)) {
                        if (minTime === null || dt < minTime) minTime = dt;
                        if (maxTime === null || dt > maxTime) maxTime = dt;
                     }
                 } else if (log.timestamp) { // Fallback to timestamp string if datetime isn't pre-parsed
                     const dt = parseTimestampFallback(log.timestamp); // Basic fallback parser
                      if (dt) {
                         if (minTime === null || dt < minTime) minTime = dt;
                         if (maxTime === null || dt > maxTime) maxTime = dt;
                      }
                 }
             } catch(e) { /* ignore date parse errors */ }
        });

        const errorRate = total > 0 ? ((errorCount / total) * 100).toFixed(1) : 0;
        const warningRate = total > 0 ? ((warningCount / total) * 100).toFixed(1) : 0;
        const avgLatency = latencyCount > 0 ? (totalLatency / latencyCount).toFixed(1) : 0.0;
        const timeSpanStr = calculateTimeSpan(minTime, maxTime);

        // 2. Update Metric DOM Elements
        metricTotalLogs.textContent = total.toLocaleString();
        metricErrorCount.textContent = errorCount.toLocaleString();
        metricErrorRate.textContent = errorRate;
        metricWarningCount.textContent = warningCount.toLocaleString();
        metricWarningRate.textContent = warningRate;
        metricTimeSpan.textContent = timeSpanStr;
        metricUniqueModules.textContent = uniqueModules.size;
        metricAvgLatency.textContent = `${avgLatency} ms`;

        // 3. Update Charts (prepare data and call Plotly)
        updateLevelPieChart();
        updateTopErrorModulesChart();
        updateLogTimelineChart();
    }

    function calculateTimeSpan(start, end) {
        if (!start || !end || start >= end) return "N/A";
        const deltaSeconds = Math.round((end - start) / 1000);
        const days = Math.floor(deltaSeconds / (3600 * 24));
        const hours = Math.floor((deltaSeconds % (3600 * 24)) / 3600);
        const minutes = Math.floor((deltaSeconds % 3600) / 60);
        const seconds = deltaSeconds % 60;

        if (days > 0) return `${days}d ${hours}h ${minutes}m`;
        if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
        if (minutes > 0) return `${minutes}m ${seconds}s`;
        return `${seconds}s`;
    }

     // Basic timestamp parser if log.datetime is not available
     function parseTimestampFallback(tsStr) {
         if (!tsStr || typeof tsStr !== 'string') return null;
         try {
             // Attempt ISO-like formats first
             let dt = new Date(tsStr);
             if (!isNaN(dt)) return dt;
             // Add more specific regex/format checks if needed for common non-ISO formats
         } catch(e) {
             return null;
         }
         return null; // Failed to parse
     }


    function updateLevelPieChart() {
        const levelCounts = logs.reduce((acc, log) => {
            const level = log.level || 'UNKNOWN';
            acc[level] = (acc[level] || 0) + 1;
            return acc;
        }, {});

        const data = [{
            values: Object.values(levelCounts),
            labels: Object.keys(levelCounts),
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: Object.keys(levelCounts).map(level => getLevelColor(level))
            },
            textinfo: 'percent+label',
            textposition: 'inside',
            hoverinfo: 'label+percent+value'
        }];

        const layout = {
            margin: { t: 30, b: 0, l: 0, r: 0 },
            height: 300,
            showlegend: false,
            template: chartThemeSelect.value // Use selected theme
        };

        Plotly.newPlot(chartLevelPie, data, layout, {responsive: true});
    }

    function updateTopErrorModulesChart() {
        const errorModules = logs
            .filter(log => log.level === 'ERROR' && log.module)
            .reduce((acc, log) => {
                acc[log.module] = (acc[log.module] || 0) + 1;
                return acc;
            }, {});

        const sortedModules = Object.entries(errorModules)
            .sort(([, a], [, b]) => a - b) // Sort ascending for horizontal bar
            .slice(-10); // Take top 10

        const data = [{
            x: sortedModules.map(([, count]) => count),
            y: sortedModules.map(([module]) => module),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: sortedModules.map(([, count]) => count), // Color by count
                colorscale: 'Reds'
            }
        }];

        const layout = {
            margin: { t: 30, b: 20, l: 100, r: 20 }, // Adjust left margin for labels
            height: 300,
            yaxis: { automargin: true }, // Adjust margin automatically
            template: chartThemeSelect.value
        };

        Plotly.newPlot(chartTopErrorModules, data, layout, {responsive: true});
    }

     function updateLogTimelineChart() {
         if (logs.length === 0) {
             Plotly.purge(chartLogTimeline); // Clear chart if no logs
             return;
         }

         // Aggregate logs by time interval (e.g., minute or hour)
         const timeData = {};
         const intervalMinutes = logs.length > 500 ? 10 : 1; // Adjust interval based on log count
         const intervalMillis = intervalMinutes * 60 * 1000;

         logs.forEach(log => {
             try {
                 const dt = log.datetime ? new Date(log.datetime) : parseTimestampFallback(log.timestamp);
                 if (dt && !isNaN(dt)) {
                    // Round down to the nearest interval
                    const timestampKey = Math.floor(dt.getTime() / intervalMillis) * intervalMillis;
                     if (!timeData[timestampKey]) {
                         timeData[timestampKey] = { INFO: 0, WARNING: 0, ERROR: 0, DEBUG: 0, PARSE_ERROR: 0, UNKNOWN: 0 };
                     }
                     const level = log.level || 'UNKNOWN';
                     timeData[timestampKey][level]++;
                 }
             } catch (e) { /* ignore */ }
         });

         const sortedTimestamps = Object.keys(timeData).map(Number).sort((a, b) => a - b);
         const xValues = sortedTimestamps.map(ts => new Date(ts));

         const levelsToPlot = ["INFO", "DEBUG", "WARNING", "ERROR", "PARSE_ERROR", "UNKNOWN"];
         const plotData = levelsToPlot.map(level => {
             return {
                 x: xValues,
                 y: sortedTimestamps.map(ts => timeData[ts][level] || 0),
                 name: level,
                 type: 'bar',
                 marker: { color: getLevelColor(level) }
             };
         }).filter(trace => trace.y.some(val => val > 0)); // Only include traces with data

         const layout = {
             barmode: 'stack',
             margin: { t: 30, b: 40, l: 40, r: 20 },
             height: 350,
             legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
             xaxis: { title: `Time (aggregated by ${intervalMinutes} min)` },
             yaxis: { title: 'Log Count' },
             template: chartThemeSelect.value
         };

         Plotly.newPlot(chartLogTimeline, plotData, layout, {responsive: true});
     }


    function getLevelColor(level) {
        level = String(level).toUpperCase();
        const colors = {
            "ERROR": "#DC3545", "WARNING": "#FFC107", "INFO": "#28A745",
            "DEBUG": "#6C757D", "PARSE_ERROR": "#FFA500", "UNKNOWN": "#adb5bd"
        };
        return colors[level] || colors["UNKNOWN"];
    }

    // --- AI Analysis ---
     function enableAnalysisButton() {
        const minLogs = 10; // Same as backend check
        if (logs.length >= minLogs) {
             runAnalysisButton.disabled = false;
             clusteringInfo.textContent = `Ready for ${logs.length} logs.`;
        } else {
             disableAnalysisButton();
        }
     }
     function disableAnalysisButton() {
         runAnalysisButton.disabled = true;
         clusteringInfo.textContent = `Requires at least 10 logs (have ${logs.length}).`;
     }

    async function runClusteringAnalysis() {
        if (logs.length < 10) {
             showStatusMessage("Not enough logs for analysis.", "error", "analysis");
             return;
        }
        showStatusMessage("Running clustering and analysis...", "loading", "analysis");
        runAnalysisButton.disabled = true;
        analysisResultsDiv.style.display = 'none'; // Hide old results

        try {
            // Send necessary log fields to backend (raw, level, module, etc.)
            // Sending all logs might be too much for large datasets.
            // A strategy might be needed to send only relevant fields or summaries.
            // For simplicity, sending essential fields of current logs:
            const logsForAnalysis = logs.map(l => ({
                raw: l.raw,
                level: l.level,
                module: l.module,
                status_code: l.status_code,
                latency: l.latency
            }));

            const nClusters = parseInt(nClustersSlider.value, 10);
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ logs: logsForAnalysis, n_clusters: nClusters })
            });

            const data = await response.json();
            if (!response.ok) {
                 throw new Error(data.error || `HTTP ${response.status}`);
            }

            showStatusMessage(data.message || "Analysis complete.", "success", "analysis");
            clusterSummaries = data.clusters_summary || [];
            errorProfiles = data.error_profiles || [];
            displayAnalysisResults(); // Render the results section

        } catch (error) {
            console.error("Analysis error:", error);
            showStatusMessage(`Analysis failed: ${error.message}`, "error", "analysis");
            clusterSummaries = null;
            errorProfiles = null;
        } finally {
             enableAnalysisButton(); // Re-enable button based on log count
        }
    }

    function displayAnalysisResults() {
        if (!clusterSummaries || clusterSummaries.length === 0) {
             analysisResultsDiv.style.display = 'none';
             return;
        }
        analysisResultsDiv.style.display = 'block';

        // Populate Cluster Explorer dropdown
        clusterSelector.innerHTML = ''; // Clear previous options
        clusterSummaries.forEach(cs => {
            const option = document.createElement('option');
            option.value = cs.cluster_id;
            option.textContent = `Cluster ${cs.cluster_id} (${cs.total_logs} logs, ${cs.error_rate}% errors)`;
            clusterSelector.appendChild(option);
        });

        // Display details for the first cluster initially
        if (clusterSummaries.length > 0) {
            clusterSelector.value = clusterSummaries[0].cluster_id;
            displaySelectedClusterDetails();
        } else {
            clusterDetailsDiv.innerHTML = '<p>No clusters found.</p>';
            analyzeClusterSummaryButton.style.display = 'none';
        }

        // Update Holistic Analysis charts
        updateClusterDistributionChart();
        updateClusterErrorRateChart();

        // Reset AI output areas
        clusterAnalysisOutput.innerHTML = '';
        holisticAnalysisOutput.innerHTML = '';

        // Activate the first nested tab (Cluster Explorer)
        switchTab(document.querySelector('.tab-link-nested[data-tab-nested="cluster-explorer"]'), nestedTabs, nestedTabContents);
    }

    function displaySelectedClusterDetails() {
         const selectedId = parseInt(clusterSelector.value, 10);
         const summary = clusterSummaries ? clusterSummaries.find(cs => cs.cluster_id === selectedId) : null;

         clusterDetailsDiv.innerHTML = ''; // Clear previous
         clusterAnalysisOutput.innerHTML = ''; // Clear previous AI analysis
         analyzeClusterSummaryButton.style.display = 'none';


         if (!summary) {
             clusterDetailsDiv.innerHTML = '<p>Select a cluster to view details.</p>';
             return;
         }

         analyzeClusterSummaryButton.style.display = 'block'; // Show button

         let detailsHtml = `
            <h5>Cluster ${summary.cluster_id} Details</h5>
            <p><strong>Total Logs:</strong> ${summary.total_logs}</p>
            <p><strong>Error Rate:</strong> ${summary.error_rate}% (${summary.error_count} errors)</p>
            <p><strong>Warning Rate:</strong> ${summary.warning_rate}% (${summary.warning_count} warnings)</p>
            <p><strong>Avg. Latency:</strong> ${summary.avg_latency?.toFixed(1)} ms</p>
            <p><strong>Top Modules:</strong></p>
            <ul>${Object.entries(summary.top_modules || {}).map(([m, c]) => `<li>${m}: ${c}</li>`).join('')}</ul>
            <p><strong>Top Status Codes:</strong></p>
            <ul>${Object.entries(summary.top_status_codes || {}).map(([s, c]) => `<li>${s}: ${c}</li>`).join('')}</ul>
            <p><strong>Sample Logs:</strong></p>
            <div class="sample-logs">
                ${(summary.sample_logs || []).map(log => `<pre><code>${escapeHtml(log)}</code></pre>`).join('')}
            </div>
         `;
         clusterDetailsDiv.innerHTML = detailsHtml;
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
     }

     async function analyzeSelectedClusterSummary() {
         const selectedId = parseInt(clusterSelector.value, 10);
         const summary = clusterSummaries ? clusterSummaries.find(cs => cs.cluster_id === selectedId) : null;
         if (!summary) return;

         analyzeClusterSummaryButton.textContent = 'Analyzing...';
         analyzeClusterSummaryButton.disabled = true;
         clusterAnalysisOutput.innerHTML = '<i>🧠 Thinking...</i>';

         try {
             const llmConfig = getLlmConfig();
             const response = await fetch('/api/llm/analyze-cluster', {
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify({
                     cluster_id: selectedId,
                     cluster_summary: summary, // Send the summary object
                     use_ollama: llmConfig.useOllama,
                     model: llmConfig.model,
                     api_key: llmConfig.apiKey,
                     ollama_url: llmConfig.ollamaUrl
                 })
             });
             const data = await response.json();
             if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);

             clusterAnalysisOutput.innerHTML = formatMarkdownBasic(data.analysis); // Basic formatting
         } catch (error) {
             console.error("Error analyzing cluster summary:", error);
             clusterAnalysisOutput.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
         } finally {
             analyzeClusterSummaryButton.textContent = 'Analyze Cluster Summary with AI';
             analyzeClusterSummaryButton.disabled = false;
         }
     }

     // Basic Markdown to HTML
     function formatMarkdownBasic(text) {
         if (!text) return '';
         let html = escapeHtml(text);
         // Simple bold, italic, code, lists - very basic
         html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
         html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
         html = html.replace(/`(.*?)`/g, '<code>$1</code>');
         html = html.replace(/^\s*[\-\*]\s+(.*)/gm, '<li>$1</li>'); // List items
         html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>'); // Wrap lists - crude
         html = html.replace(/\n/g, '<br>'); // Newlines
         return html;
     }


     function updateClusterDistributionChart() {
         if (!clusterSummaries) return;
         const data = [{
             x: clusterSummaries.map(cs => `Cluster ${cs.cluster_id}`),
             y: clusterSummaries.map(cs => cs.total_logs),
             type: 'bar',
             marker: { color: clusterSummaries.map(cs => cs.cluster_id), colorscale: 'Viridis' } // Color by cluster ID
         }];
         const layout = {
             title: 'Log Distribution by Cluster',
             margin: { t: 30, b: 30, l: 40, r: 20 }, height: 300,
             xaxis: { type: 'category' }, yaxis: { title: 'Log Count'},
             template: chartThemeSelect.value
        };
         Plotly.newPlot(chartClusterDist, data, layout, {responsive: true});
     }

     function updateClusterErrorRateChart() {
          if (!clusterSummaries) return;
         const data = [{
             x: clusterSummaries.map(cs => `Cluster ${cs.cluster_id}`),
             y: clusterSummaries.map(cs => cs.error_rate),
             type: 'bar',
             marker: { color: clusterSummaries.map(cs => cs.error_rate), colorscale: 'Reds' } // Color by error rate
         }];
         const layout = {
             title: 'Error Rate (%) by Cluster',
             margin: { t: 30, b: 30, l: 40, r: 20 }, height: 300,
             xaxis: { type: 'category' }, yaxis: { title: 'Error Rate (%)', range: [0, 100] },
             template: chartThemeSelect.value
         };
         Plotly.newPlot(chartClusterErrRate, data, layout, {responsive: true});
     }

     async function runHolisticAnalysis() {
        if (!clusterSummaries || logs.length === 0) {
            holisticAnalysisOutput.innerHTML = '<p class="error-message">Need logs and cluster analysis results first.</p>';
            return;
        }
        runHolisticButton.textContent = 'Analyzing...';
        runHolisticButton.disabled = true;
        holisticAnalysisOutput.innerHTML = '<i>🧠 Thinking...</i>';

         // Calculate overall summary (similar to dashboard metrics)
         const total = logs.length;
         const errorCount = logs.filter(l=>l.level==='ERROR').length;
         const warningCount = logs.filter(l=>l.level==='WARNING').length;
         const errorRate = total > 0 ? ((errorCount / total) * 100).toFixed(1) : 0;
         const warningRate = total > 0 ? ((warningCount / total) * 100).toFixed(1) : 0;
         const logSummaryData = { total_logs: total, error_count: errorCount, warning_count: warningCount, error_rate: errorRate, warning_rate: warningRate };


         try {
            // TODO: Add backend endpoint /api/llm/holistic
            // For now, simulate or show message
            /*
             const llmConfig = getLlmConfig();
             const response = await fetch('/api/llm/holistic', { // Needs implementing
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify({
                     log_summary: logSummaryData,
                     clusters_summary: clusterSummaries,
                     use_ollama: llmConfig.useOllama, model: llmConfig.model,
                     api_key: llmConfig.apiKey, ollama_url: llmConfig.ollamaUrl
                 })
             });
             const data = await response.json();
             if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
             holisticAnalysisOutput.innerHTML = formatMarkdownBasic(data.analysis);
            */
             holisticAnalysisOutput.innerHTML = formatMarkdownBasic(`**Holistic Analysis (Placeholder)**\n\nAnalysis would compare overall stats:\n*   Total Logs: ${total}\n*   Error Rate: ${errorRate}%\n\nWith cluster summaries:\n${clusterSummaries.map(cs => `*   Cluster ${cs.cluster_id}: ${cs.total_logs} logs, ${cs.error_rate}% errors`).join('\n')}\n\n(Backend endpoint /api/llm/holistic needs implementation)`);

         } catch (error) {
             console.error("Error running holistic analysis:", error);
             holisticAnalysisOutput.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
         } finally {
             runHolisticButton.textContent = 'Generate Holistic Analysis with AI';
             runHolisticButton.disabled = false;
         }
     }



    // --- LLM Settings ---
    function updateLlmProviderView() {
        const useOllama = document.querySelector('input[name="llmProvider"][value="ollama"]').checked;
        ollamaSettings.style.display = useOllama ? 'block' : 'none';
        openrouterSettings.style.display = useOllama ? 'none' : 'block';
    }

    function getLlmConfig() {
        const useOllama = document.querySelector('input[name="llmProvider"][value="ollama"]').checked;
        let config = { useOllama: useOllama };
        if (useOllama) {
            config.model = ollamaModelInput.value.trim();
            config.ollamaUrl = ollamaUrlInput.value.trim();
        } else {
            config.model = openrouterModelSelect.value;
            config.apiKey = openrouterKeyInput.value.trim();
        }
        return config;
    }


    // --- Tab Switching ---
    function switchTab(clickedTab, tabSet, contentSet) {
        tabSet.forEach(tab => tab.classList.remove('active'));
        contentSet.forEach(content => content.classList.remove('active'));

        clickedTab.classList.add('active');
        const targetTabId = clickedTab.getAttribute('data-tab') || clickedTab.getAttribute('data-tab-nested');
        document.getElementById(targetTabId).classList.add('active');
    }

    // --- Initial setup ---
    updateLlmProviderView(); // Show settings for default LLM provider

}); // End DOMContentLoaded