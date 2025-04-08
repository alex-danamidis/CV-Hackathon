const video = document.getElementById('video');
const gestureLabel = document.getElementById('gesture-label');

// Load ONNX model
let session;
const loadModel = async () => {
  try {
    // Path should be relative to your Flask static folder
    session = await ort.InferenceSession.create('/static/asl_model.onnx');
    console.log('Model loaded');
  } catch (err) {
    console.error('Error loading model:', err);
    gestureLabel.textContent = 'Error loading model';
  }
};

// Set up webcam
const setupCamera = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error('Error accessing webcam: ', err);
    gestureLabel.textContent = 'Error accessing webcam';
  }
};

// Initialize MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});
hands.onResults(onResults);

// Create a canvas to capture cropped hand image
const offscreenCanvas = document.createElement('canvas');
const offscreenCtx = offscreenCanvas.getContext('2d');

// Create a canvas to draw the bounding box and hand on the video
const boundingBoxCanvas = document.createElement('canvas');
const boundingBoxCtx = boundingBoxCanvas.getContext('2d');
document.body.appendChild(boundingBoxCanvas); // Append canvas to body for drawing

// Handle results from MediaPipe Hands
async function onResults(results) {
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    gestureLabel.textContent = 'No Hand Detected';
    return;
  }

  const handLandmarks = results.multiHandLandmarks[0];
  const boundingBox = getBoundingBox(handLandmarks);

  if (!boundingBox) {
    gestureLabel.textContent = 'No Hand Detected';
    return;
  }

  // Draw bounding box on the main canvas (video overlay)
  drawBoundingBox(boundingBox);

  // Crop the hand from the video
  const { x, y, width, height } = boundingBox;
  offscreenCanvas.width = width;
  offscreenCanvas.height = height;
  offscreenCtx.drawImage(video, x, y, width, height, 0, 0, width, height);

  const tfImage = tf.browser.fromPixels(offscreenCanvas);

  // Resize the cropped hand to 64x64
  const resized = tf.image.resizeBilinear(tfImage, [64, 64]);
  const normalized = resized.div(255.0);
  const batched = normalized.expandDims(0);

  // Create tensor
  const inputTensor = new ort.Tensor('float32', batched.dataSync(), [1, 3, 64, 64]);
  tfImage.dispose();
  resized.dispose();
  normalized.dispose();
  batched.dispose();

  // Predict using ONNX model
  try {
    const feeds = { input: inputTensor };
    const resultsONNX = await session.run(feeds);

    const output = resultsONNX.output.data;
    const maxConfidence = Math.max(...output);
    const predictedIndex = output.indexOf(maxConfidence);

    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const confidenceThreshold = 0.7;

    if (maxConfidence < confidenceThreshold) {
      gestureLabel.textContent = 'No Gesture';
    } else {
      gestureLabel.textContent = alphabet[predictedIndex] || 'Detecting...';
    }

    inputTensor.dispose();
  } catch (error) {
    console.error('Prediction error:', error);
    gestureLabel.textContent = 'Error predicting';
  }
}

// Get bounding box from landmarks
function getBoundingBox(landmarks) {
  const xs = landmarks.map(p => p.x * video.videoWidth);
  const ys = landmarks.map(p => p.y * video.videoHeight);

  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);

  const width = maxX - minX;
  const height = maxY - minY;

  return {
    x: Math.max(0, Math.floor(minX)),
    y: Math.max(0, Math.floor(minY)),
    width: Math.floor(width),
    height: Math.floor(height)
  };
}

// Draw bounding box on the video canvas
function drawBoundingBox(boundingBox) {
  boundingBoxCtx.clearRect(0, 0, boundingBoxCanvas.width, boundingBoxCanvas.height);  // Clear the previous drawing
  boundingBoxCtx.strokeStyle = 'red';
  boundingBoxCtx.lineWidth = 2;
  boundingBoxCtx.setLineDash([5, 3]);  // Optional: dashed line
  boundingBoxCtx.strokeRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);
}

// Set up webcam and start gesture detection loop
const detectGesture = async () => {
  if (!session) return;

  await hands.send({ image: video });
  requestAnimationFrame(detectGesture);
};

window.onload = async () => {
  await setupCamera();
  await loadModel();
  detectGesture();
};
