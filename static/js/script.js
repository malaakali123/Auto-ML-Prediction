// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('fileName');
const targetSelection = document.getElementById('target-selection');
const targetInput = document.getElementById('targetColumn');
const startBtn = document.getElementById('startBtn');
const uploadSection = document.getElementById('upload-section');
const loadingSection = document.getElementById('loading-section');
const resultSection = document.getElementById('result-section');
const problemTypeDisplay = document.getElementById('problem-type');
const bestModelDisplay = document.getElementById('best-model-name');
const bestScoreDisplay = document.getElementById('best-model-score');
const leaderboardBody = document.querySelector('#leaderboard-table tbody');
const loadingText = document.getElementById('loading-text');

let selectedFile = null;

// Event Listeners for Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (file.name.endsWith('.csv')) {
        selectedFile = file;
        fileNameDisplay.textContent = `Selected: ${file.name}`;
        fileNameDisplay.style.color = '#10b981';
        targetSelection.classList.remove('hidden');
        checkStartButton();
    } else {
        alert('Please upload a valid CSV file.');
        fileNameDisplay.textContent = 'Invalid file type';
        fileNameDisplay.style.color = '#ef4444';
    }
}

targetInput.addEventListener('input', checkStartButton);

function checkStartButton() {
    if (selectedFile && targetInput.value.trim().length > 0) {
        startBtn.disabled = false;
    } else {
        startBtn.disabled = true;
    }
}

startBtn.addEventListener('click', async () => {
    if (!selectedFile || !targetInput.value.trim()) return;

    // UI Transition
    uploadSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');

    // Rotating loading text
    const loadingSteps = [
        "Analyzing Dataset Structure...",
        "Cleaning Missing Values...",
        "Detecting Problem Type...",
        "Preprocessing Features...",
        "Training Machine Learning Models...",
        "Tuning Hyperparameters...",
        "Finalizing Leaderboard...",
        "Generating Visualizations..."
    ];
    let stepIndex = 0;
    const interval = setInterval(() => {
        if (stepIndex < loadingSteps.length) {
            loadingText.textContent = loadingSteps[stepIndex];
            stepIndex++;
        }
    }, 2000);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target', targetInput.value.trim());

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        clearInterval(interval);

        if (response.ok) {
            showResults(data);
        } else {
            alert('Error: ' + (data.error || 'Something went wrong'));
            location.reload();
        }

    } catch (error) {
        clearInterval(interval);
        console.error('Error:', error);
        alert('An unexpected error occurred.');
        location.reload();
    }
});

function showResults(data) {
    loadingSection.classList.add('hidden');
    resultSection.classList.remove('hidden');

    // Update Summary
    problemTypeDisplay.textContent = data.problem_type;
    bestModelDisplay.textContent = data.best_model;
    bestScoreDisplay.textContent = (data.best_score * 100).toFixed(2) + '%';

    // Populate Leaderboard
    leaderboardBody.innerHTML = '';
    data.leaderboard.forEach((row, index) => {
        const tr = document.createElement('tr');
        // Add badges for top 3
        let rankDisplay = index + 1;
        if (index === 0) rankDisplay = '🥇';
        if (index === 1) rankDisplay = '🥈';
        if (index === 2) rankDisplay = '🥉';

        tr.innerHTML = `
            <td>${rankDisplay}</td>
            <td style="font-weight: 500">${row.model_name}</td>
            <td>${(row.score * 100).toFixed(2)}%</td>
            <td style="font-family: monospace; font-size: 0.9em">${JSON.stringify(row.metrics).replace(/{|}|"/g, '')}</td>
        `;
        leaderboardBody.appendChild(tr);
    });

    // Render Charts
    renderComparisonChart(data.leaderboard);
    renderFeatureChart(data.feature_importance);
}

function renderComparisonChart(leaderboard) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    const labels = leaderboard.map(m => m.model_name);
    const data = leaderboard.map(m => m.score * 100);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Model Accuracy (%)',
                data: data,
                backgroundColor: [
                    'rgba(245, 158, 11, 0.8)',  // Gold
                    'rgba(251, 191, 36, 0.8)',  // Lighter Gold
                    'rgba(217, 119, 6, 0.8)',   // Dark Amber
                    'rgba(255, 255, 255, 0.2)', // White transparent
                    'rgba(148, 163, 184, 0.5)'  // Gray
                ],
                borderColor: 'rgba(245, 158, 11, 0.3)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function renderFeatureChart(avgFeatures) {
    if (!avgFeatures || Object.keys(avgFeatures).length === 0) return;

    const ctx = document.getElementById('featureChart').getContext('2d');

    // Sort features
    const sorted = Object.entries(avgFeatures).sort((a, b) => b[1] - a[1]).slice(0, 7); // Top 7
    const labels = sorted.map(i => i[0]);
    const values = sorted.map(i => i[1]);

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    '#f59e0b', '#fbbf24', '#d97706', '#ffffff',
                    '#94a3b8', '#475569', '#1e293b'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'right', labels: { color: '#94a3b8' } }
            }
        }
    });
}
