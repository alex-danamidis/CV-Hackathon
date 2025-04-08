const asl_classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ".split("");  // Ensure this matches the model's classes
const video = document.getElementById('video');
const gestureLabel = document.getElementById('gesture-label');
const canvas = document.getElementById('bounding-box-canvas');
const ctx = canvas.getContext('2d');

let session;
let inputName = "";
let detecting = false; // Prevent duplicate processing

// Load the ONNX model
const loadModel = async () => {
    try {
        session = await ort.InferenceSession.create("/static/asl_model.onnx");
        inputName = session.inputNames ? session.inputNames[0] : session._modelMeta.inputNames[0];
        console.log("✅ ONNX Model Loaded:", inputName);
    } catch (err) {
        console.error("❌ Error loading model:", err);
        gestureLabel.textContent = "Error loading model";
    }
};

// Set up webcam
const setupCamera = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, frameRate: { ideal: 30, max: 30 } }
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("❌ Error accessing webcam: ", err);
        gestureLabel.textContent = "Error accessing webcam";
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

hands.onResults(async (results) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
            drawLandmarks(ctx, landmarks);
        }
        gestureLabel.textContent = "Hand detected";
        
        // Run prediction ONLY IF NOT already processing
        if (!detecting) {
            detecting = true;
            await predictGesture();
            detecting = false;
        }
    } else {
        gestureLabel.textContent = "No hand detected";
    }
});

// Draw landmarks on canvas
const drawLandmarks = (ctx, landmarks) => {
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    for (let i = 0; i < landmarks.length; i++) {
        const x = landmarks[i].x * canvas.width;
        const y = landmarks[i].y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
    }
};

// Convert image to tensor with proper preprocessing
function preprocessImage(video) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const width = 64;  // Model expects 64x64 input
    const height = 64;

    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(video, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    const input = new Float32Array(3 * width * height);

    // Normalize to match the training normalization
    for (let i = 0; i < width * height; i++) {
        input[i] = (data[i * 4] / 255.0 - 0.5) / 0.5;  // R
        input[i + width * height] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5;  // G
        input[i + 2 * width * height] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5;  // B
    }

    return new ort.Tensor("float32", input, [1, 3, width, height]);
}

// Apply softmax to convert logits to probabilities
function softmax(logits) {
    const exp = logits.map(x => Math.exp(x));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(x => x / sum);
}

async function predictGesture() {
    if (!session) return;

    try {
        const tensor = preprocessImage(video);
        const feeds = { [inputName]: tensor };
        const resultsONNX = await session.run(feeds);
        const outputArray = Array.from(resultsONNX.output.data);

        console.log("Raw output:", outputArray);  // Log the raw logits for debugging

        // Apply softmax if necessary
        const probabilities = softmax(outputArray);

        // Find the predicted class index
        const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
        gestureLabel.textContent = `Detected: ${asl_classes[predictedIndex] || "Unknown"}`;
    } catch (error) {
        console.error("❌ Prediction error:", error);
        gestureLabel.textContent = "Prediction Error";
    }
}

// Start detection loop
const detectGesture = async () => {
    await predictGesture();
    setTimeout(detectGesture, 300); // Lower refresh rate to reduce lag
};

window.onload = async () => {
    await setupCamera();
    await loadModel();
    const camera = new Camera(video, {
        onFrame: async () => {
            await hands.send({ image: video });
        },
        width: 640,
        height: 480
    });
    camera.start();
};
