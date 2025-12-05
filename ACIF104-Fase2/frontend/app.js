
document.addEventListener('DOMContentLoaded', () => {

    // Elements
    const modelSelect = document.getElementById('model-select');
    const featuresContainer = document.getElementById('features-container');
    const predictForm = document.getElementById('predict-form');
    const resultArea = document.getElementById('result-area');
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section-content');

    // Navigation
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            const targetId = link.getAttribute('data-target');
            sections.forEach(sec => sec.classList.add('d-none'));
            document.getElementById(targetId).classList.remove('d-none');
        });
    });

    // Load Initial Data
    loadModels();
    loadFeatures();
    loadMetrics();
    checkHealth();

    // Fetch Models
    async function loadModels() {
        try {
            const res = await fetch('/models');
            const data = await res.json();
            modelSelect.innerHTML = '<option disabled selected>Selecciona un modelo</option>';
            data.models.forEach(model => {
                const opt = document.createElement('option');
                opt.value = model;
                opt.textContent = model;
                modelSelect.appendChild(opt);
            });
        } catch (error) {
            console.error('Error loading models:', error);
            modelSelect.innerHTML = '<option disabled>Error cargando modelos</option>';
        }
    }

    // Fetch Features and Build Form
    async function loadFeatures() {
        try {
            const res = await fetch('/features');
            const data = await res.json();
            featuresContainer.innerHTML = '';

            if (data.features.length === 0) {
                featuresContainer.innerHTML = '<p class="text-warning">No features metadata found.</p>';
                return;
            }

            data.features.forEach(feat => {
                const div = document.createElement('div');
                div.className = 'mb-2';
                div.innerHTML = `
                    <label class="form-label small mb-1">${feat}</label>
                    <input type="number" step="any" class="form-control form-control-sm" name="${feat}" required>
                `;
                featuresContainer.appendChild(div);
            });
        } catch (error) {
            console.error('Error loading features:', error);
            featuresContainer.innerHTML = '<p class="text-danger">Error cargando features.</p>';
        }
    }

    // Handle Prediction
    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const model = modelSelect.value;
        if (!model || model.includes('Selecciona')) {
            alert("Por favor selecciona un modelo.");
            return;
        }

        const formData = new FormData(predictForm);
        const features = {};
        formData.forEach((value, key) => {
            features[key] = parseFloat(value);
        });

        resultArea.innerHTML = '<div class="spinner-border text-primary" role="status"></div><p class="mt-2">Prediciendo...</p>';

        try {
            const res = await fetch(`/predict?model=${model}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: features })
            });

            if (!res.ok) throw new Error(await res.text());

            const data = await res.json();

            resultArea.innerHTML = `
                <h2 class="display-4 text-primary">${data.prediction.toFixed(2)}</h2>
                <p class="text-muted">Unidades Demandadas</p>
                <div class="mt-3 badge bg-secondary">Modelo: ${data.model_used}</div>
                <div class="badge bg-info">Latencia: ${data.latency_ms.toFixed(2)} ms</div>
            `;

        } catch (error) {
            console.error(error);
            resultArea.innerHTML = `
                <div class="alert alert-danger">Error: ${error.message}</div>
            `;
        }
    });

    // Load Metrics Tables
    async function loadMetrics() {
        // ML
        try {
            const res = await fetch('/metrics?type=ml');
            const data = await res.json();
            const tableBody = document.querySelector('#ml-metrics-table tbody');
            if (Array.isArray(data)) {
                tableBody.innerHTML = data.map(row => `
                    <tr>
                        <td>${row.Model}</td>
                        <td>${row.MSE.toFixed(2)}</td>
                        <td>${row.RMSE.toFixed(2)}</td>
                        <td>${row.MAE.toFixed(2)}</td>
                        <td>${row.R2.toFixed(4)}</td>
                        <td>${row.MAPE.toFixed(2)}%</td>
                    </tr>
                `).join('');
            }
        } catch (e) { console.warn(e); }

        // DL
        try {
            const res = await fetch('/metrics?type=dl');
            const data = await res.json();
            const tableBody = document.querySelector('#dl-metrics-table tbody');
            if (Array.isArray(data)) {
                tableBody.innerHTML = data.map(row => `
                    <tr>
                        <td>${row.Model}</td>
                        <td>${row.MSE.toFixed(2)}</td>
                        <td>${row.RMSE.toFixed(2)}</td>
                        <td>${row.MAE.toFixed(2)}</td>
                        <td>${row.R2.toFixed(4)}</td>
                        <td>${row.MAPE.toFixed(2)}%</td>
                    </tr>
                `).join('');
            } else {
                tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No DL metrics available yet.</td></tr>';
            }
        } catch (e) { console.warn(e); }

        // Clustering
        try {
            const res = await fetch('/metrics?type=clustering');
            const data = await res.json();
            const tableBody = document.querySelector('#clustering-metrics-table tbody');
            if (Array.isArray(data)) {
                tableBody.innerHTML = data.map((row, i) => `
                    <tr>
                        <td>${row.K || i + 2}</td>
                        <td>${row.Inertia ? row.Inertia.toFixed(2) : '-'}</td>
                        <td>${row.Silhouette ? row.Silhouette.toFixed(4) : '-'}</td>
                    </tr>
                `).join('');
            }
        } catch (e) { console.warn(e); }
    }

    // Check Health
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            document.getElementById('api-status').textContent = data.status === 'ok' ? 'ONLINE' : 'OFFLINE';
        } catch (e) {
            document.getElementById('api-status').textContent = 'OFFLINE';
            document.getElementById('api-status').parentElement.parentElement.classList.replace('bg-success', 'bg-danger');
        }
    }
});
