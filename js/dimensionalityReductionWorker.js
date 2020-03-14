let that = this;

// Import UMAP and t-SNE libraries.
this.window = this;
importScripts('umap-js.js');
importScripts('tsne.min.js');

async function performTsne(latentSpaceData, mnistData, returnArr, n=1000) {
    // Code adapted from example code: https://github.com/scienceai/tsne-js
    postText('Initializing t-SNE...');

    // Set model parameters and initialize model.
    let model = new TSNE({
        dim: 2,
        perplexity: 5.0,
        earlyExaggeration: 4.0,
        learningRate: 100.0,
        nIter: 200,
        metric: 'euclidean'
    });
    model.init({
        data: latentSpaceData,
        type: 'dense'
    });

    // Set callback functions for point and iteration updates.
    model.on('progressData', function (data) {
        for (let i = 0; i < data.length; i++) {
            returnArr[i]['point'] = data[i];
        }
        postMessage({type: 'points', data: returnArr});;
    });
    model.on('progressIter', function(iter) {
        postText('t-SNE Iteration: ' + iter[0]);
    });

    // Run t-SNE
    let out = model.run();

    // Parse return array
    let output = model.getOutput();
    for (let i = 0; i < output.length; i++) {
        returnArr[i]['point'] = output[i];
    }

    return returnArr;
}

async function performUmap(latentSpaceData, mnistData, returnArr, n=3000) {
    // Code adapted from example code: https://github.com/PAIR-code/umap-js
    postText('Initializing UMAP...');

    // Create UMAP.
    that.umap = new UMAP();

    // Initialize and step through UMAP.
    let epochs = that.umap.initializeFit(latentSpaceData);
    for (let i = 0; i < epochs; i++) {
        that.umap.step();
        postText('UMAP Iteration ' + i);

        // Every 10 steps send update back.
        if (i % 10 === 0) {
            let currentEmbedding = await umap.getEmbedding();
            for (let i = 0; i < currentEmbedding.length; i++) {
                returnArr[i]['point'] = currentEmbedding[i];
            }

            // Post update
            postMessage({type: 'points', data: returnArr});
        }
    }

    // Get return array
    postText('UMAP Finished');
    let currentEmbedding = await umap.getEmbedding();
    for (let i = 0; i < currentEmbedding.length; i++) {
        returnArr[i]['point'] = currentEmbedding[i];
    }

    return returnArr;
}

// Post status update back to website.
function postText(text) {
    postMessage({type: 'text', data:text});
}

// Handle incoming posted messages.
onmessage = async function(request) {
    console.log('Dim reduction worker received command.');
    let type = request.data.type;
    let values = request.data.values;

    // Handle request on data.
    let returnArr;
    if (type === 'UMAP') {
        returnArr = await performUmap(values[0], values[1], values[2]);
    } else if (type === 't-SNE') {
        returnArr = await performTsne(values[0], values[1], values[2]);
    }

    // Return dimensionally reduced points.
    console.log('Dim reduction worker returning values.');
    postMessage({type: 'points', data: returnArr});
};