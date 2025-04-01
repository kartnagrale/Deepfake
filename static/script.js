let realImages = [];
let fakeImages = [];
let selectedModel = null;

// Upload real images
document.getElementById("realImages").addEventListener("change", function (event) {
    realImages = event.target.files;
});

// Upload fake images
document.getElementById("fakeImages").addEventListener("change", function (event) {
    fakeImages = event.target.files;
});

// Upload model
document.getElementById("modelUpload").addEventListener("change", function (event) {
    selectedModel = event.target.files[0];
});

// Train Model
function trainModel() {
    if (realImages.length === 0 || fakeImages.length === 0) {
        alert("Please upload both real and fake images.");
        return;
    }

    let formData = new FormData();
    for (let img of realImages) formData.append("real_images", img);
    for (let img of fakeImages) formData.append("fake_images", img);

    fetch("http://127.0.0.1:5000/train", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("trainingResults").innerHTML = `
            <p>Model trained successfully!</p>
            <p>Accuracy: ${data.accuracy}%</p>
            <p>F1 Score: ${data.f1_score}</p>
            <a href="/download_model" download>
                <button>Download Model</button>
            </a>
        `;
        })
        .catch(error => alert("Error training model."));
}

// Detect Image
function detectImage() {
    let imageFile = document.getElementById("testImage").files[0];

    if (!selectedModel || !imageFile) {
        alert("Please upload a model and an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", imageFile);

    fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = `Result: ${data.result}`;
        })
        .catch(error => alert("Error processing image."));
}
