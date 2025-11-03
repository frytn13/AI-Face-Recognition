/**
 * ================================================================================================
 * Proyek AI Face Recognition - Final Version (Enhanced Landmark Colors)
 * ================================================================================================
 */

// --- Ambil Elemen HTML ---
const video = document.getElementById('video');
const loadingScreen = document.getElementById('loading-screen');
const registerButton = document.getElementById('registerButton');
const personNameInput = document.getElementById('personName');
const personImagesInput = document.getElementById('personImages');
const toggleLandmarksCheckbox = document.getElementById('toggleLandmarks');

// --- Variabel Global ---
let labeledFaceDescriptors = [];
let faceMatcher = null;
let showLandmarks = false;

// ================================================================================================
// BAGIAN 1: INISIALISASI & SETUP
// ================================================================================================

async function initializeApp() {
    const modelsLoaded = await loadModels();
    if (modelsLoaded) {
        await loadHybridFaceData();
        updateFaceMatcher();
        loadingScreen.classList.add('hidden');
        startVideo();
    }
}

async function loadModels() {
    console.log("Memulai pemuatan model AI...");
    try {
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
            faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
            faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
            faceapi.nets.faceExpressionNet.loadFromUri('/models'),
            faceapi.nets.ageGenderNet.loadFromUri('/models')
        ]);
        console.log("✅ Model AI berhasil dimuat!");
        return true;
    } catch (err) {
        console.error("❌ Gagal memuat model AI:", err);
        return false;
    }
}

function startVideo() {
    console.log("Mencoba memulai webcam...");
    navigator.mediaDevices.getUserMedia({ video: {} })
        .then(stream => {
            video.srcObject = stream;
            console.log("✅ Webcam berhasil diakses.");
        })
        .catch(err => console.error("❌ Error mengakses webcam:", err));
}

toggleLandmarksCheckbox.addEventListener('change', (event) => {
    showLandmarks = event.target.checked;
    console.log(`Tampilkan Garis Wajah (Landmarks): ${showLandmarks}`);
});

// ================================================================================================
// BAGIAN 2: HYBRID LEARNING (localStorage + Manual Folder)
// ================================================================================================

async function loadHybridFaceData() {
    const localData = loadDescriptorsFromLocalStorage();
    const manualData = await loadDescriptorsFromFolders();
    const combinedLabels = new Map();

    manualData.forEach(ld => combinedLabels.set(ld.label, ld));
    localData.forEach(ld => {
        if (!combinedLabels.has(ld.label)) {
            combinedLabels.set(ld.label, ld);
        }
    });

    labeledFaceDescriptors = Array.from(combinedLabels.values());
    console.log("💾 Data wajah hybrid berhasil dimuat dan digabungkan.");
}

async function loadDescriptorsFromFolders() {
    console.log("Mencari data wajah dari folder manual...");
    const manualDescriptors = [];
    const labels = ['Budi', 'Siti']; // <-- Ganti atau tambahkan nama di sini

    for (const label of labels) {
        const descriptions = [];
        try {
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`./labeled_images/${label}/${i}.jpg`);
                const detection = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
                if (detection) {
                    descriptions.push(detection.descriptor);
                }
            }
            if (descriptions.length > 0) {
                manualDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptions));
                console.log(`✅ Data manual untuk '${label}' berhasil dimuat.`);
            }
        } catch (error) {
            console.warn(`⚠️ Folder atau gambar untuk '${label}' tidak ditemukan.`);
        }
    }
    return manualDescriptors;
}

// ================================================================================================
// BAGIAN 3: FUNGSI TAMPILAN & VISUALISASI
// ================================================================================================

/**
 * Menggambar landmark wajah dengan kustomisasi (warna dan ketebalan).
 * Menggunakan skema warna yang lebih menarik dan beragam.
 */
function drawCustomLandmarks(detections, canvas) {
    detections.forEach(detection => {
        const landmarks = detection.landmarks;
        const ctx = canvas.getContext('2d');

        // Definisi warna yang lebih menarik
        const jawColor = '#FFD700';       // Gold
        const mouthColor = '#FF69B4';     // Hot Pink
        const noseColor = '#1E90FF';      // Dodger Blue
        const eyeColor = '#32CD32';       // Lime Green
        const eyebrowColor = '#FFA500';   // Orange

        const lineWidth = 2; // Ketebalan garis

        faceapi.draw.drawContour(ctx, landmarks.getJawOutline(), { color: jawColor, lineWidth: lineWidth });
        faceapi.draw.drawContour(ctx, landmarks.getMouth(), { color: mouthColor, lineWidth: lineWidth });
        faceapi.draw.drawContour(ctx, landmarks.getNose(), { color: noseColor, lineWidth: lineWidth });
        faceapi.draw.drawContour(ctx, landmarks.getLeftEye(), { color: eyeColor, lineWidth: lineWidth });
        faceapi.draw.drawContour(ctx, landmarks.getRightEye(), { color: eyeColor, lineWidth: lineWidth });
        faceapi.draw.drawContour(ctx, landmarks.getLeftEyeBrow(), { color: eyebrowColor, lineWidth: lineWidth });
        faceapi.draw.drawContour(ctx, landmarks.getRightEyeBrow(), { color: eyebrowColor, lineWidth: lineWidth });
    });
}


// ================================================================================================
// BAGIAN 4: LOOP DETEKSI REAL-TIME
// ================================================================================================

video.addEventListener('play', () => {
    console.log("Video mulai diputar, memulai loop deteksi.");
    const canvas = faceapi.createCanvasFromMedia(video);
    document.querySelector('.video-wrapper').append(canvas);
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        resizedDetections.forEach(detection => {
            const box = detection.detection.box;
            const age = Math.round(detection.age);
            const gender = detection.gender === 'male' ? 'Laki-laki' : 'Perempuan';
            const maxExpression = Object.keys(detection.expressions).reduce((a, b) => detection.expressions[a] > detection.expressions[b] ? a : b);

            let label = "Tidak Dikenal";
            if (faceMatcher) {
                const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
                if (bestMatch.label !== 'unknown') {
                    label = `${bestMatch.label} (${Math.round((1 - bestMatch.distance) * 100)}%)`;
                }
            }

            new faceapi.draw.DrawBox(box, {
                label: label,
                boxColor: 'rgba(80, 227, 194, 0.7)',
                drawLabelOptions: { fontColor: 'white', fontSize: 18, padding: 5, backgroundColor: 'rgba(0, 0, 0, 0.5)' }
            }).draw(canvas);

            new faceapi.draw.DrawTextField(
                [`${gender}, ~${age} tahun`, `Emosi: ${maxExpression}`],
                { x: box.bottomLeft.x, y: box.bottomLeft.y + 5 },
                { fontColor: 'white', fontSize: 16, padding: 5, backgroundColor: 'rgba(0, 0, 0, 0.5)' }
            ).draw(canvas);

            if (showLandmarks) {
                drawCustomLandmarks(resizedDetections, canvas);
            }
        });
    }, 150);
});

// ================================================================================================
// BAGIAN 5: FUNGSI PENDAFTARAN & LOCALSTORAGE
// ================================================================================================

registerButton.addEventListener('click', async () => {
    const name = personNameInput.value.trim();
    const images = personImagesInput.files;
    if (!name || images.length === 0) { return alert("Harap masukkan nama dan pilih setidaknya satu foto."); }

    loadingScreen.classList.remove('hidden');
    loadingScreen.querySelector('p').textContent = `🧠 Mempelajari wajah ${name}...`;

    const descriptions = [];
    for (let i = 0; i < images.length; i++) {
        try {
            const img = await faceapi.bufferToImage(images[i]);
            const detection = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
            if (detection) { descriptions.push(detection.descriptor); }
        } catch (error) { console.error("❌ Gagal memproses gambar:", error); }
    }

    if (descriptions.length > 0) {
        const existingPerson = labeledFaceDescriptors.find(d => d.label === name);
        if (existingPerson) {
            existingPerson.descriptors.push(...descriptions);
        } else {
            labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(name, descriptions));
        }
        saveDescriptorsToLocalStorage();
        updateFaceMatcher();
        alert(`✅ Wajah ${name} berhasil dipelajari dan disimpan!`);
    } else {
        alert(`⚠️ Gagal mendaftarkan ${name}. Pastikan foto jelas.`);
    }

    personNameInput.value = '';
    personImagesInput.value = '';
    loadingScreen.classList.add('hidden');
    loadingScreen.querySelector('p').textContent = `Memuat Model AI, mohon tunggu...`;
});

function saveDescriptorsToLocalStorage() {
    const dataToSave = labeledFaceDescriptors.map(ld => ({
        label: ld.label,
        descriptors: ld.descriptors.map(d => Array.from(d))
    }));
    localStorage.setItem('face_recognition_data', JSON.stringify(dataToSave));
    console.log("💾 Deskriptor wajah disimpan ke localStorage.");
}

function loadDescriptorsFromLocalStorage() {
    const data = localStorage.getItem('face_recognition_data');
    if (data) {
        const parsedData = JSON.parse(data);
        console.log("💾 Deskriptor wajah berhasil dimuat dari localStorage.");
        return parsedData.map(pd => {
            const descriptors = pd.descriptors.map(d => new Float32Array(d));
            return new faceapi.LabeledFaceDescriptors(pd.label, descriptors);
        });
    }
    return [];
}

function updateFaceMatcher() {
    if (labeledFaceDescriptors.length > 0) {
        faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
        console.log("🔄️ FaceMatcher berhasil diperbarui.");
    }
}

// ================================================================================================
// JALANKAN APLIKASI
// ================================================================================================
initializeApp();