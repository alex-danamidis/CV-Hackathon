const video = document.getElementById("video");
const gestureLabel = document.getElementById("gesture-label");

// ASL classes mapping (ensure this matches your training labels)
const ASL_CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".split("");

// Load the ONNX model
async function loadModel() {
    try {
        console.log("Loading model...");
        const session = await ort.InferenceSession.create("/model/asl_model.onnx"); // Adjusted path
        console.log("Model loaded successfully.");
        return session;
    } catch (error) {
        console.error("Error loading model:", error);
        gestureLabel.innerText = "Error loading model";
    }
}

// Start video stream
async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        console.log("Video stream started.");
    } catch (error) {
        console.error("Error starting video stream:", error);
        gestureLabel.innerText = "Error starting video stream";
    }
}

// Detect ASL alphabet gestures in real-time
async function detectGesture(session) {
    try {
        const webcam = await tf.data.webcam(video);

        while (true) {
            const img = await webcam.capture();
            const resizedImg = tf.image.resizeBilinear(img, [64, 64]);
            const normalizedImg = resizedImg.div(255.0).expandDims(0); // Normalize the image

            // Convert from HWC (Height, Width, Channels) -> CHW (Channels, Height, Width) for ONNX
            const transposedImg = normalizedImg.transpose([0, 3, 1, 2]);
            const tensorData = transposedImg.dataSync();  // Get tensor data

            // Create ONNX tensor and feed it to the model
            const tensor = new ort.Tensor("float32", tensorData, [1, 3, 64, 64]);

            // Run model inference
            const feeds = { input: tensor };
            const results = await session.run(feeds);

            console.log("ONNX Model Output:", results.output.data);

            if (results.output && results.output.data.length > 0) {
                // Get index of the highest probability class
                const predictedIndex = results.output.data.indexOf(Math.max(...results.output.data));

                // Get ASL letter corresponding to the index
                const detectedGesture = ASL_CLASSES[predictedIndex] || "Unknown";

                // Update UI
                gestureLabel.innerText = `Detected Gesture: ${detectedGesture}`;
                console.log("Detected Gesture:", detectedGesture);
            } else {
                console.warn("No gesture detected.");
                gestureLabel.innerText = "No gesture detected";
            }

            img.dispose(); // Dispose of the image tensor to free memory
            await tf.nextFrame();
        }
    } catch (error) {
        console.error("Error in model inference:", error);
        gestureLabel.innerText = "Inference error";
    }
}

// Initialize everything
(async () => {
    await startVideo();
    const session = await loadModel();
    if (session) {
        detectGesture(session);
    }
})();
