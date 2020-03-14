// Default settings
let that = this;
this.currentDataDes = 'MNIST';
console.log('Tensorflow Backend: ' + tf.getBackend());

// Paths to models
const dim2InfPath = './models/cvae-mnist-dim2/infModel/model.json';
const dim2GenPath = './models/cvae-mnist-dim2/genModel/model.json';
const dim10InfPath = './models/cvae-mnist-dim10/infModel/model.json';
const dim10GenPath = './models/cvae-mnist-dim10/genModel/model.json';
const dim50InfPath = './models/cvae-mnist-dim50/infModel/model.json';
const dim50GenPath = './models/cvae-mnist-dim50/genModel/model.json';

const dim2InfFashionPath = './models/cvae-fashion-dim2/infModel/model.json';
const dim2GenFashionPath = './models/cvae-fashion-dim2/genModel/model.json';
const dim10InfFashionPath = './models/cvae-fashion-dim10/infModel/model.json';
const dim10GenFashionPath = './models/cvae-fashion-dim10/genModel/model.json';
const dim50InfFashionPath = './models/cvae-fashion-dim50/infModel/model.json';
const dim50GenFashionPath = './models/cvae-fashion-dim50/genModel/model.json';

////////////////////////////////////////////
////////////////////////////////////////////
//////////// Update Functions //////////////
////////////////////////////////////////////
////////////////////////////////////////////

async function updateProgress(text) {
    d3.select('#reductionProgress').text(text);
}

async function updateLabels(d) {
    d3.select('#xValue').text('X: ' + d.x.toFixed(2));
    d3.select('#yValue').text('Y: ' + d.y.toFixed(2));
}

async function clearInterpolate() {
    let row = d3.select('#interpolateRow').classed('hideIt', true);
    row.html('');
}

async function updateInterpolate(point1, point2, steps=22) {
    let row = d3.select('#interpolateRow').classed('hideIt', false);
    row.html('');

    // Add the initial point image.
    if (point1 !== undefined) {
        row.append('canvas').attr('id', 'selectedPoint1Canvas').attr('width', 84).attr('height', 84).style('border', '2px solid red');

        let tensor = await that.upscaleTensorToDisplay(point1.data.reshape([28,28]), 3);
        tf.browser.toPixels(tensor, document.getElementById('selectedPoint1Canvas'));
    }

    // Add all interpolated images in between.
    if (point1 !== undefined && point2 !== undefined) {
        let interpolatedMeans = [];
        for (let i = 0; i < point1.mean.length; i++) {
            let tmpArr = [];
            let tmp = (point2.mean[i] - point1.mean[i]) / steps;
            interpolatedMeans.push([]);

            let val = point1.mean[i];
            for (let j = 0; j < steps; j++) {
                interpolatedMeans[i].push(val);
                val += tmp
            }
        }

        for (let i = 0; i < steps; i++) {
            let arr = [];
            for (let j = 0; j < interpolatedMeans.length; j++) {
                arr.push(interpolatedMeans[j][i]);
            }

            let tensor = tf.tensor(arr).reshape([1, interpolatedMeans.length]);
            row.append('canvas')
                .attr('id', 'interpolateCanvas' + i)
                .attr('width', 84)
                .attr('height', 84)
                .style('padding', '1px');

            let decoded = that.selectedModel.decode(tensor);
            tensor = await that.upscaleTensorToDisplay(decoded.reshape([28,28]), 3);
            tf.browser.toPixels(tensor, document.getElementById('interpolateCanvas' + i));
        }
    }

    // Add the second selected point image.
    if (point2 !== undefined) {
        row.append('canvas').attr('id', 'selectedPoint2Canvas').attr('width', 84).attr('height', 84).style('border', '2px solid blue');

        let tensor = await that.upscaleTensorToDisplay(point2.data.reshape([28,28]), 3);
        tf.browser.toPixels(tensor, document.getElementById('selectedPoint2Canvas'));
    }
}

async function updateImage(d) {
    // Handle if model isn't loaded/ready yet.
    if (that.selectedModel === undefined) {
        // Do nothing.
        return;
    }

    let x = d.x;
    let y = d.y;

    let output = that.selectedModel.decode(tf.tensor([[x, y]])).reshape([28,28]);
    let tensor = await that.upscaleTensorToDisplay(output);
    tf.browser.toPixels(tensor, document.getElementById('reconstructedCanvas'));

    // Clear other canvas
    d3.select('#originalCanvas').attr('width', 84);
}

async function updateImages(d) {
    // Display original image
    let tensor = await that.upscaleTensorToDisplay(d.data.reshape([28,28]), 3);
    tf.browser.toPixels(tensor, document.getElementById('originalCanvas'));

    // Display recreated image
    let encoded = that.selectedModel.encode(d.data);
    let decoded = that.selectedModel.decode(encoded[0]);
    tensor = await that.upscaleTensorToDisplay(decoded.reshape([28,28]), 3);
    tf.browser.toPixels(tensor, document.getElementById('reconstructedCanvas'));
}

function updateDisplayedSubsetData() {
    let in0 = Math.floor(Math.random() * 2800);
    let in1 = Math.floor(Math.random() * 2800);
    let in2 = Math.floor(Math.random() * 2800);
    let in3 = Math.floor(Math.random() * 2800);
    that.displaySubset([in0, in1, in2, in3]);
}

async function upscaleTensorToDisplay(tensor, scale=3) {
    let tensorArr = await tensor.array();
    let upscaledArr = [];
    let width = 28 * scale;
    for (let i = 0; i < width; i++) {
        upscaledArr.push([]);
        for (let j = 0; j < width; j++) {
            upscaledArr[i].push({});
        }
    }

    for (let i = 0; i < width; i+=scale) {
        for (let j = 0; j < width; j+=scale) {
            for (let k = 0; k < scale; k++) {
                let xPixelIndex = parseInt(i/scale);
                let yPixelIndex = parseInt(j/scale);
                for (let l = 0; l < scale; l++) {
                    upscaledArr[i + l][j + k] = tensorArr[xPixelIndex][yPixelIndex];
                }
            }
        }
    }

    return tf.tensor(upscaledArr).reshape([width, width])
}

// Display 4 original and reconstructed images with the current data and model.
async function displaySubset(indices) {
    for (let i = 0; i < 4; i++) {
        let index = indices[i];

        // Display original image
        let tensor = await that.upscaleTensorToDisplay(that.currentData[index].data.reshape([28,28]), 3);
        tf.browser.toPixels(tensor, document.getElementById('mnistCanvas' + i));

        // Display recreated image
        let encoded = that.selectedModel.encode(this.currentData[index].data);
        let decoded = that.selectedModel.decode(encoded[0]);
        tensor = await that.upscaleTensorToDisplay(decoded.reshape([28,28]), 3);
        tf.browser.toPixels(tensor, document.getElementById('mnistCanvasRec' + i));
    }
}

// Encode the current data to get the latent space vectors.
async function encodeData() {
    that.returnArr = [];
    that.latentSpaceData = [];
    that.updateProgress('Encoding data...');
    let encoded = await that.selectedModel.encode(that.tensors);
    let means = await encoded[0].array();
    let stds = await encoded[1].array();
    for (let i = 0; i < 3000; i++) {
        let mean = means[i];
        let std = stds[i];
        that.latentSpaceData.push(mean.flat());
        that.returnArr.push({
            data: that.currentData[i].data,
            label: that.currentData[i].label,
            id: that.currentData[i].id,
            mean: mean,
            point: mean.flat(),
            std: std.flat()
        });
    }
}

// Create the PointChart to display image points.
that.pointChart = new PointChart(updateImage, updateImages, updateLabels, updateInterpolate);
that.pointChart.updateChart();

// Create the WebWorker for dimensionality reduction.
this.dimReductionWorker = new Worker('js/dimensionalityReductionWorker.js');
this.dimReductionWorker.onmessage = function(e) {
    let object = e.data;

    if (object.type === 'text') {
        d3.select('#reductionProgress').text(object.data);
    } else if (object.type === 'points') {
        for (let i = 0; i < that.returnArr.length; i++) {
            that.returnArr[i].point = object.data[i].point;
        }

        that.pointChart.updateDisplayedPoints(that.returnArr);
    }
};

// Load models from memory.
Promise.all([
    tf.loadLayersModel(dim2InfPath),
    tf.loadLayersModel(dim2GenPath)
]).then(function(models) {
    that.updateProgress('Loaded first model...');
    that.vae2 = new Vae(models[0], models[1], 2);

    // Load other models on different thread.
    Promise.all([
        tf.loadLayersModel(dim10InfPath),
        tf.loadLayersModel(dim10GenPath),
        tf.loadLayersModel(dim50InfPath),
        tf.loadLayersModel(dim50GenPath),
        tf.loadLayersModel(dim2InfFashionPath),
        tf.loadLayersModel(dim2GenFashionPath),
        tf.loadLayersModel(dim10InfFashionPath),
        tf.loadLayersModel(dim10GenFashionPath),
        tf.loadLayersModel(dim50InfFashionPath),
        tf.loadLayersModel(dim50GenFashionPath)
    ]).then(function(models) {
        that.vae10 = new Vae(models[0], models[1], 10);
        that.vae50 = new Vae(models[2], models[3], 50);

        that.fashionVae2 = new Vae(models[4], models[5], 2);
        that.fashionVae10 = new Vae(models[6], models[7], 10);
        that.fashionVae50 = new Vae(models[8], models[9], 50);
    });

    // Set selected model
    that.selectedModel = that.vae2;

    // Load mnist fashion data.
    d3.csv('./data/mnist-fashion.csv').then(async function(d) {
        that.mnistFashionData = [];
        that.mnistFashionArr = [];
        for (let i = 0; i < 3000; i++) {
            let tmpData = {
                label: d[i].label,
                id: 'data-' + i
            };

            let tmpArr = [];
            for (let j = 0; j < 784; j++) {
                tmpArr.push(parseInt(d[i]['pixel'+j])/255.0);
            }

            tmpData['data'] = tf.tensor(tmpArr).reshape([1, 28, 28, 1]);
            tmpData['dataArr'] = tmpArr;
            that.mnistFashionData.push(tmpData);
            that.mnistFashionArr.push(tmpArr);
        }
    });

    // Load MNIST data
    d3.csv('./data/mnist.csv').then(async function(d) {
        that.updateProgress('Loading data...');
        that.mnistData = [];
        that.mnistArr = [];
        for (let i = 0; i < 3000; i++) {
            let tmpData = {
                label: d[i].label,
                id: 'data-' + i
            };

            let tmpArr = [];
            for (let j = 0; j < 784; j++) {
                tmpArr.push(parseInt(d[i]['pixel'+j])/255.0);
            }

            tmpData['data'] = tf.tensor(tmpArr).reshape([1, 28, 28, 1]);
            tmpData['dataArr'] = tmpArr;
            that.mnistData.push(tmpData);
            that.mnistArr.push(tmpArr);
        }

        // Set current tensors and data.
        that.tensors = tf.tensor(that.mnistArr).reshape([3000, 28, 28, 1]);
        that.currentData = that.mnistData.slice();

        // Display subset of data points.
        that.displaySubset([20, 30, 40, 50]);

        // Encode data with selected model and data.
        await encodeData();

        // Display points in 2d vae
        that.returnArr2Dim = that.returnArr.slice();
        that.updateProgress('Finished loading');
        let pointsToDisplay = that.returnArr2Dim;
        that.pointChart.updateDisplayedPoints(pointsToDisplay);
    });
});

///////////////////////////////////////////
///////////////////////////////////////////
/////////////// On Click //////////////////
///////////////////////////////////////////
///////////////////////////////////////////

// Setup display subset button.
d3.select('#mnistButton').on('click', function() {
    that.updateDisplayedSubsetData();
});

// Hide dim reduction setting (since latent space dim is initially 2)
d3.select('#dimReductionDiv').classed('hideIt', true);

// Handle button clicks for model/data.
d3.select('#cvaeMnist').on('click', function() {
    d3.select('#modelDropdown').text('CVAE - MNIST');
});
d3.select('#cvaeFashion').on('click', function() {
    d3.select('#modelDropdown').text('CVAE - Fashion');
});

// Handle button clicks for latent dimension size.
d3.select('#latent2').on('click', function() {
    d3.select('#dimReductionDropdown').text('UMAP');
    d3.select('#dimReductionDiv').classed('hideIt', true);
    d3.select('#latentDimensionSizeDropdown').text('2');
});
d3.select('#latent10').on('click', function() {
    d3.select('#dimReductionDropdown').text('UMAP');
    d3.select('#dimReductionDiv').classed('hideIt', false);
    d3.select('#latentDimensionSizeDropdown').text('10');
});
d3.select('#latent50').on('click', function() {
    d3.select('#dimReductionDropdown').text('UMAP');
    d3.select('#dimReductionDiv').classed('hideIt', false);
    d3.select('#latentDimensionSizeDropdown').text('50');
});

// Handle button clicks for dimensionality reduction method.
d3.select('#UMAP').on('click', function() {
    d3.select('#dimReductionDropdown').text('UMAP');
});
d3.select('#t-SNE').on('click', function() {
    d3.select('#dimReductionDropdown').text('t-SNE');
});

// Handle update configuration button.
d3.select('#updateButton').on('click', async function() {
    // Clear interpolation div and remove points.
    clearInterpolate();
    that.pointChart.updateDisplayedPoints([]);

    // Handle a change of data.
    let modelData = d3.select('#modelDropdown').text();
    if (modelData === 'CVAE - Fashion' && that.currentDataDes === 'MNIST') {
        that.tensors = tf.tensor(that.mnistFashionArr).reshape([3000, 28, 28, 1]);
        that.currentData = that.mnistFashionData.slice();
        that.currentDataDes = 'MNIST Fashion';
    } else if (modelData === 'CVAE - MNIST' && that.currentDataDes !== 'MNIST') {
        that.tensors = tf.tensor(that.mnistArr).reshape([3000, 28, 28, 1]);
        that.currentData = that.mnistData.slice();
        that.currentDataDes = 'MNIST';
    }

    let dimReduction = d3.select('#dimReductionDropdown').text();
    let latentSize = parseInt(d3.select('#latentDimensionSizeDropdown').text());

    if (latentSize === 2) {
        // Set model and update latent dim.
        that.pointChart.updateLatentDim(2);
        if (that.currentDataDes === 'MNIST') {
            that.selectedModel = that.vae2;
        } else {
            that.selectedModel = that.fashionVae2;
        }

        // Update displayed subset.
        that.updateDisplayedSubsetData();

        // Encode the selected data with the selected model.
        await encodeData();

        // Update displayed points.
        that.pointChart.updateDisplayedPoints(that.returnArr);
    } else if (latentSize === 10) {
        // Set model and update latent dim.
        that.pointChart.updateLatentDim(10);
        if (that.currentDataDes === 'MNIST') {
            that.selectedModel = that.vae10;
        } else {
            that.selectedModel = that.fashionVae10;
        }

        // Update displayed subset.
        that.updateDisplayedSubsetData();

        // Encode the selected data with the selected model.
        await encodeData();

        // Create the configuration for dimensionality reduction on the web worker.
        let config = {
            type: 'UMAP',
            values: [that.latentSpaceData, that.currentData, that.returnArr]
        };
        if (dimReduction === 't-SNE') {
            config.type = 't-SNE';
        }

        // Start dimensionality reduction web worker.
        that.dimReductionWorker.postMessage(config);
    } else if (latentSize === 50) {
        // Set model and update latent dim.
        that.pointChart.updateLatentDim(50);
        if (that.currentDataDes === 'MNIST') {
            that.selectedModel = that.vae50;
        } else {
            that.selectedModel = that.fashionVae50;
        }

        // Update displayed subset.
        that.updateDisplayedSubsetData();

        // Encode the selected data with the selected model.
        await encodeData();

        // Create the configuration for the dimensionality reduction on the web worker.
        let config = {
            type: 'UMAP',
            values: [that.latentSpaceData, that.currentData, that.returnArr]
        };
        if (dimReduction === 't-SNE') {
            config.type = 't-SNE';
        }

        // Start dimensionality reduction web worker.
        that.dimReductionWorker.postMessage(config);
    }
});