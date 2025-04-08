const video = document.getElementById('video');
const gestureLabel = document.getElementById('gesture-label');

let session;

const loadModel = async () => {
  try {
    // session = await ort.InferenceSession.create(window.location.origin + "/static/asl_model.onnx");
    session = await ort.InferenceSession.create("/static/asl_model.onnx");
 // Fixed path
    console.log("✅ ONNX Model Loaded");
  } catch (err) {
    console.error("❌ Error loading model:", err);
    gestureLabel.textContent = "Error loading model";
    gestureLabel.textContent = (err);
  }
};

// Set up webcam
const setupCamera = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error("❌ Error accessing webcam: ", err);
    gestureLabel.textContent = "Error accessing webcam";
  }
};

// Convert image to tensor
function preprocessImage(video) {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, 64, 64);

  const imageData = ctx.getImageData(0, 0, 64, 64);
  const data = imageData.data;
  const input = new Float32Array(3 * 64 * 64);

  for (let i = 0; i < 64 * 64; i++) {
    input[i] = data[i * 4] / 255.0; // R
    input[i + 64 * 64] = data[i * 4 + 1] / 255.0; // G
    input[i + 2 * 64 * 64] = data[i * 4 + 2] / 255.0; // B
  }

  return new ort.Tensor("float32", input, [1, 3, 64, 64]);
}

// Predict from image
async function predictGesture() {
  if (!session) return;

  const tensor = preprocessImage(video);
  try {
    const feeds = { input: tensor };
    const resultsONNX = await session.run(feeds);
    const output = resultsONNX.output.data;

    const predictedIndex = output.indexOf(Math.max(...output));
    const aslClasses = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ".split("");

    gestureLabel.textContent = aslClasses[predictedIndex] || "Unknown";
  } catch (error) {
    console.error("❌ Prediction error:", error);
    gestureLabel.textContent = "Prediction Error";
  }
}

// Start prediction loop
const detectGesture = async () => {
  await predictGesture();
  requestAnimationFrame(detectGesture);
};

window.onload = async () => {
  await setupCamera();
  await loadModel();
  detectGesture();
};
