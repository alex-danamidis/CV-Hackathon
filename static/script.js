const video = document.getElementById('video');
const gestureLabel = document.getElementById('gesture-label');

// Load the model
async function loadModel() {
    const model = await tf.loadLayersModel('/model/model.json');
    return model;
}

// Start video stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

// Detect ASL alphabet gestures in real-time
video.addEventListener('loadeddata', async () => {
    const model = await loadModel();
    const webcam = await tf.data.webcam(video);
    while (true) {
        const img = await webcam.capture();
        const predictions = model.predict(img.expandDims(0));
        const gesture = predictions.argMax(-1).dataSync()[0];
        gestureLabel.innerText = `Detected Gesture: ${gesture}`;
        img.dispose();
        await tf.nextFrame();
    }
});
