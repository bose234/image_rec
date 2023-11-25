const express = require('express');
const bodyParser = require('body-parser');
const { createCanvas, loadImage } = require('canvas');
const tf = require('@tensorflow/tfjs');
const cors = require('cors');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const axios = require('axios');

const app = express();
const port = 3000;
app.use(cors())

app.use(bodyParser.json());
app.use(express.json()); 
app.use(express.urlencoded({ extended: true }));

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

app.post('/', async (req, res) => {
  try {
    let inputImage;
    if (req.body.imageUrl) {
      // Process image from URL
      const imageUrl = req.body.imageUrl;
      const { data: imageBuffer } = await axios.get(imageUrl, { responseType: 'arraybuffer' });

      const canvas = createCanvas();
      const ctx = canvas.getContext('2d');

      const img = await loadImage(Buffer.from(imageBuffer));
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0, img.width, img.height);

      inputImage = tf.browser.fromPixels(canvas, 3);
    } else if (req.body.imageData) {
      // Process image from camera
      const base64Data = req.body.imageData.replace(/^data:image\/png;base64,/, '');
      const imageBuffer = Buffer.from(base64Data, 'base64');

      const canvas = createCanvas();
      const ctx = canvas.getContext('2d');

      const img = await loadImage(imageBuffer);
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0, img.width, img.height);

      inputImage = tf.browser.fromPixels(canvas, 3);
    } else {
      throw new Error('Image URL or image data is missing in the request body.');
    }

    const model = await cocoSsd.load();

    const predictions = await model.detect(inputImage);

    const processedObjects = predictions.map((prediction, index) => {
      return {
        objectname : prediction.class,
      };
    });

    res.json(processedObjects);
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
