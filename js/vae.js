// Adapted from original construction of models in python located here: https://research.google.com/seedbank/seed/convolutional_vae
// https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb
class Vae {
    constructor(infModel, genModel, latentDim) {
        this.latentDim = latentDim;
        this.inferenceModel = infModel;
        this.generativeModel = genModel;
    }

    // Encode image and output mean and std vectors.
    encode(input) {
        return tf.split(this.inferenceModel.predict(input), 2, 1);
    }

    // Decode latent space vector back to image.
    decode(z) {
        return tf.sigmoid(this.generativeModel.predict(z));
    }
}