import * as tf from "@tensorflow/tfjs-core";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tfjsWasm from "@tensorflow/tfjs-backend-wasm";

tfjsWasm.setWasmPaths(
	`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`
);

function isMobile() {
	const isAndroid = /Android/i.test(navigator.userAgent);
	const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
	return isAndroid || isiOS;
}

function distance(a, b) {
	return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

const video = document.getElementById("display");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");
const model = faceLandmarksDetection.SupportedPackages.mediapipeFacemesh;
const mobile = isMobile();
const VIDEO_SIZE = 500;

const img = new Image();
img.src = "./pngwing.com.png";

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = "#32EEDB";
const RED = "#FF2C35";
const BLUE = "#157AB3";

let videoWidth, videoHeight;

const setupCamera = async () => {
	const stream = await navigator.mediaDevices.getUserMedia({
		audio: false,
		video: {
			facingMode: "user",
			width: mobile ? undefined : VIDEO_SIZE,
			height: mobile ? undefined : VIDEO_SIZE,
		},
	});
	video.srcObject = stream;

	return new Promise((resolve) => {
		video.onloadedmetadata = () => {
			resolve(video);
		};
	});
};

const drawItem = ({
	leftEyeIris,
	rightEyeIris,
	leftEyebrowUpper,
	rightEyebrowUpper,
	leftEyebrowLower,
	rightEyebrowLower,
	leftEyeLower3,
	rightEyeLower3,
}) => {
	let _path = [];

	_path.push(leftEyeIris[0]); // center
	_path.push(rightEyeIris[0]); // center

	_path.push(leftEyebrowLower[0]); // start
	_path.push(rightEyebrowLower[0]); // end

	_path.push(leftEyebrowUpper[4]); // top start
	_path.push(rightEyebrowUpper[4]); // top end

	_path.push(leftEyeLower3[4]); // bottom start
	_path.push(rightEyeLower3[4]); // bottom end

	ctx.fillStyle = GREEN;

	_path.forEach((p) => {
		ctx.beginPath();
		ctx.arc(p[0], p[1], 5, 0, 2 * Math.PI);
		ctx.fill();
	});

	const width = distance(leftEyebrowLower[0], rightEyebrowLower[0]);
	const height = distance(leftEyebrowUpper[4], leftEyeLower3[4]);

	ctx.strokeStyle = GREEN;
	ctx.save();

	ctx.beginPath();
	ctx.translate(leftEyebrowLower[0][0], leftEyebrowUpper[4][1]);
	ctx.rotate(
		Math.atan2(
			rightEyeIris[0][1] - leftEyebrowLower[0][1],
			rightEyeIris[0][0] - leftEyebrowLower[0][0]
		)
	);
	ctx.drawImage(img, -15, 15, width + 30, height);
	ctx.fill();

	ctx.restore();
};

const renderPrediction = async (detector) => {
	// clear context

	const predict = (
		await detector.estimateFaces({
			input: video,
			returnTensors: false,
			flipHorizontal: true,
			predictIrises: true,
		})
	)[0];

	ctx.clearRect(0, 0, videoWidth, videoHeight);

	ctx.fillStyle = GREEN;

	if (predict) {
		console.log(predict);
		drawItem(predict.annotations);
	}

	requestAnimationFrame(() => renderPrediction(detector));
};

const init = async () => {
	await tf.setBackend("wasm");
	await setupCamera();
	video.play();

	videoWidth = video.videoWidth;
	videoHeight = video.videoHeight;
	video.width = videoWidth;
	video.height = videoHeight;

	canvas.width = videoWidth;
	canvas.height = videoHeight;

	const detector = await faceLandmarksDetection.load(model, {
		maxFaces: 1,
	});

	renderPrediction(detector);
};

document.addEventListener("DOMContentLoaded", () => init());
