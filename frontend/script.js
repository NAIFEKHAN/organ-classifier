const API_BASE_URL = "https://boolean-solution-indices-rolled.trycloudflare.com";

const viewport = document.getElementById("viewport");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const dropzoneContent = document.getElementById("dropzoneContent");
const sweep = document.getElementById("sweep");
const fileMeta = document.getElementById("fileMeta");
const modalityMeta = document.getElementById("modalityMeta");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMessage = document.getElementById("errorMessage");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const resultEmpty = document.getElementById("resultEmpty");
const resultBody = document.getElementById("resultBody");
const predictedName = document.getElementById("predictedName");
const predictedConfidence = document.getElementById("predictedConfidence");
const probabilityList = document.getElementById("probabilityList");
const endpointLabel = document.getElementById("endpointLabel");

endpointLabel.textContent = `${API_BASE_URL}/predict`;

let currentFile = null;

function setStatus(state, label) {
  statusPill.dataset.state = state;
  statusText.textContent = label;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  setStatus("error", "ERROR");
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function loadFile(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg", "image/jpg"].includes(file.type)) {
    showError("Please upload a PNG or JPEG image.");
    return;
  }

  clearError();
  currentFile = file;

  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewImage.alt = `Preview of uploaded scan: ${file.name}`;
  previewImage.hidden = false;
  dropzoneContent.style.display = "none";
  viewport.dataset.empty = "false";

  fileMeta.textContent = `${file.name.toUpperCase()} · ${formatBytes(file.size)}`;
  modalityMeta.textContent = "MODALITY: AUTO";

  clearBtn.disabled = false;
  analyzeBtn.disabled = false;
  setStatus("ready", "READY TO ANALYZE");

  resultBody.hidden = true;
  resultEmpty.hidden = false;
  resultEmpty.textContent = "";
  resultEmpty.innerHTML = "<p>Scan loaded. Click \u201cAnalyze scan\u201d to classify it.</p>";
}

function resetViewer() {
  currentFile = null;
  fileInput.value = "";
  previewImage.src = "";
  previewImage.hidden = true;
  dropzoneContent.style.display = "";
  viewport.dataset.empty = "true";
  fileMeta.textContent = "NO FILE LOADED";
  modalityMeta.textContent = "MODALITY: AUTO";
  clearBtn.disabled = true;
  analyzeBtn.disabled = true;
  clearError();
  setStatus("idle", "READY");
  sweep.classList.remove("active");

  resultBody.hidden = true;
  resultEmpty.hidden = false;
  resultEmpty.innerHTML = "<p>Upload a scan and run analysis to see the predicted organ here.</p>";
}

// --- File input + drag and drop ---

fileInput.addEventListener("change", (e) => {
  loadFile(e.target.files[0]);
});

["dragover", "dragenter"].forEach((evt) => {
  viewport.addEventListener(evt, (e) => {
    e.preventDefault();
    viewport.classList.add("dragover");
  });
});

["dragleave", "dragend"].forEach((evt) => {
  viewport.addEventListener(evt, () => {
    viewport.classList.remove("dragover");
  });
});

viewport.addEventListener("drop", (e) => {
  e.preventDefault();
  viewport.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

clearBtn.addEventListener("click", resetViewer);

// --- Analyze ---

analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  clearError();
  setStatus("analyzing", "ANALYZING…");
  sweep.classList.add("active");
  analyzeBtn.disabled = true;
  clearBtn.disabled = true;
  resultBody.hidden = true;
  resultEmpty.hidden = false;
  resultEmpty.innerHTML = "<p>Running inference…</p>";

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "The server couldn't process this image.");
    }

    renderResult(data);
    setStatus("ready", "ANALYSIS COMPLETE");
  } catch (err) {
    const message = err instanceof TypeError
      ? `Couldn't reach the backend at ${API_BASE_URL}. Is it running?`
      : err.message;
    showError(message);
    resultEmpty.innerHTML = "<p>Upload a scan and run analysis to see the predicted organ here.</p>";
  } finally {
    sweep.classList.remove("active");
    analyzeBtn.disabled = false;
    clearBtn.disabled = false;
  }
});

function renderResult(data) {
  resultEmpty.hidden = true;
  resultBody.hidden = false;

  predictedName.textContent = data.prediction;
  predictedConfidence.textContent = `${data.confidence.toFixed(1)}% CONFIDENCE`;

  probabilityList.innerHTML = "";
  data.all_probabilities.forEach((item) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${item.organ}</span>
      <span class="prob-track"><span class="prob-fill" style="width:0%"></span></span>
      <span class="prob-value">${item.confidence.toFixed(1)}%</span>
    `;
    probabilityList.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".prob-fill").style.width = `${item.confidence}%`;
    });
  });
}
