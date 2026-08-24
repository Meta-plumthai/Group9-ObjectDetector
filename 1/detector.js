import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let objectDetector;
let runningMode = "IMAGE";
let scoreThreshold = 0.5;
let maxResults = 3;
let delegate = "GPU";
let selectedModel = "efficientdet_lite0";

const statusEl = document.getElementById("status");
const videoDisplay = document.getElementById("videoDisplay");
const imageDisplay = document.getElementById("imageDisplay");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

const btnImage = document.getElementById("btnImage");
const btnWebcam = document.getElementById("btnWebcam");
const imageUpload = document.getElementById("imageUpload");
const uploadBtnLabel = document.getElementById("uploadBtnLabel");

const thresholdInput = document.getElementById("scoreThreshold");
const thresholdVal = document.getElementById("thresholdVal");
const maxResultsInput = document.getElementById("maxResults");
const maxResultsVal = document.getElementById("maxResultsVal");
const delegateSelect = document.getElementById("delegateSelect");
const modelSelect = document.getElementById("modelSelect");

const doneText = document.getElementById("doneText");
const inferenceTimeEl = document.getElementById("inferenceTime");

// 1. โหลดโมเดล
async function initDetector() {
  if (statusEl) statusEl.innerText = "Loading...";
  const modelUrl = `https://storage.googleapis.com/mediapipe-models/object_detector/${selectedModel}/float16/1/${selectedModel}.tflite`;
  
  try {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelUrl,
        delegate: delegate
      },
      scoreThreshold: scoreThreshold,
      maxResults: maxResults,
      runningMode: runningMode
    });
    if (statusEl) statusEl.innerText = "Ready";
  } catch (err) {
    if (statusEl) statusEl.innerText = "Error";
    console.error(err);
  }
}
initDetector();

// 2. วาด Bounding Boxes
function drawBoundingBoxes(detections, element) {
  const naturalW = element.videoWidth || element.naturalWidth || element.clientWidth;
  const naturalH = element.videoHeight || element.naturalHeight || element.clientHeight;

  const displayW = element.clientWidth;
  const displayH = element.clientHeight;

  canvas.width = displayW;
  canvas.height = displayH;
  ctx.clearRect(0, 0, displayW, displayH);

  if (!detections || naturalW === 0) return;

  const scaleX = displayW / naturalW;
  const scaleY = displayH / naturalH;

  detections.forEach((detection) => {
    const x = detection.boundingBox.originX * scaleX;
    const y = detection.boundingBox.originY * scaleY;
    const w = detection.boundingBox.width * scaleX;
    const h = detection.boundingBox.height * scaleY;

    const category = detection.categories[0];
    const labelText = `${category.categoryName} (${Math.round(category.score * 100)}%)`;

    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.font = "600 13px 'Prompt', sans-serif";
    const textW = ctx.measureText(labelText).width;
    const boxH = 24;

    ctx.fillStyle = "#06b6d4";
    ctx.fillRect(x, y, textW + 14, boxH);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(labelText, x + 7, y + 16);
  });
}

// 3. ประมวลผลภาพ
async function processImage() {
  if (!objectDetector || imageDisplay.style.display === "none" || !imageDisplay.src) return;

  if (runningMode !== "IMAGE") {
    runningMode = "IMAGE";
    await objectDetector.setOptions({ runningMode: "IMAGE" });
  }

  const startTime = performance.now();
  const result = objectDetector.detect(imageDisplay);
  const duration = performance.now() - startTime;

  doneText.innerText = `Done in ${Math.round(duration)}ms`;
  inferenceTimeEl.innerText = duration.toFixed(2);

  drawBoundingBoxes(result.detections, imageDisplay);
}

// 4. การจัดการไฟล์รูปภาพ
imageUpload.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    imageDisplay.src = event.target.result;
    imageDisplay.style.display = "block";
    videoDisplay.style.display = "none";
    imageDisplay.onload = () => processImage();
  };
  reader.readAsDataURL(file);
});

// 5. สลับโหมด Live Cam / Image
btnWebcam.addEventListener("click", async () => {
  btnWebcam.classList.add("active");
  btnImage.classList.remove("active");
  uploadBtnLabel.style.display = "none";

  imageDisplay.style.display = "none";
  videoDisplay.style.display = "block";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoDisplay.srcObject = stream;
    videoDisplay.onloadedmetadata = () => {
      videoDisplay.play();
      renderWebcam();
    };
  } catch (err) {
    alert("ไม่สามารถเปิดกล้องได้");
  }
});

btnImage.addEventListener("click", () => {
  btnImage.classList.add("active");
  btnWebcam.classList.remove("active");
  uploadBtnLabel.style.display = "inline-flex";

  videoDisplay.style.display = "none";
  imageDisplay.style.display = "block";

  if (videoDisplay.srcObject) {
    videoDisplay.srcObject.getTracks().forEach(track => track.stop());
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  processImage();
});

async function renderWebcam() {
  if (btnWebcam.classList.contains("active") && objectDetector) {
    if (runningMode !== "VIDEO") {
      runningMode = "VIDEO";
      await objectDetector.setOptions({ runningMode: "VIDEO" });
    }

    const startTime = performance.now();
    const result = objectDetector.detectForVideo(videoDisplay, startTime);
    const duration = performance.now() - startTime;

    doneText.innerText = `Done in ${Math.round(duration)}ms`;
    inferenceTimeEl.innerText = duration.toFixed(2);

    drawBoundingBoxes(result.detections, videoDisplay);
    requestAnimationFrame(renderWebcam);
  }
}

// 6. Controls Event Listeners
thresholdInput.addEventListener("input", async (e) => {
  scoreThreshold = parseFloat(e.target.value);
  thresholdVal.innerText = scoreThreshold.toFixed(2);
  if (objectDetector) {
    await objectDetector.setOptions({ scoreThreshold: scoreThreshold });
    if (btnImage.classList.contains("active")) processImage();
  }
});

maxResultsInput.addEventListener("input", async (e) => {
  maxResults = parseInt(e.target.value);
  maxResultsVal.innerText = maxResults;
  if (objectDetector) {
    await objectDetector.setOptions({ maxResults: maxResults });
    if (btnImage.classList.contains("active")) processImage();
  }
});

delegateSelect.addEventListener("change", async (e) => {
  delegate = e.target.value;
  await initDetector();
  if (btnImage.classList.contains("active")) processImage();
});

modelSelect.addEventListener("change", async (e) => {
  selectedModel = e.target.value;
  await initDetector();
  if (btnImage.classList.contains("active")) processImage();
});