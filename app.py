from flask import Flask, request, redirect, render_template, session
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

UPLOADS = os.path.join('static', 'uploads')

app = Flask(__name__)
app.config['UPLOADS'] = UPLOADS

app.secret_key = "super secret key"



imageForWork = None

def load_model():
  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
  model.eval()
  return model

def make_transparent_foreground(pic, mask):
  # split the image into channels
  b, g, r = cv2.split(np.array(pic).astype('uint8'))
  # add an alpha channel with and fill all with transparent pixels (max 255)
  a = np.ones(mask.shape, dtype='uint8') * 255
  # merge the alpha channel back
  alpha_im = cv2.merge([b, g, r, a], 4)
  # create a transparent background
  bg = np.zeros(alpha_im.shape)
  # setup the new mask
  new_mask = np.stack([mask, mask, mask, mask], axis=2)
  # copy only the foreground color pixels from the original image where mask is set
  foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

  return foreground

def remove_background(model, input_file):
  input_image = Image.open(input_file)
  new_height = 500
  new_width  = new_height * input_image.width / input_image.height
  new_size = (int(new_width), int(new_height))
  resized_image = input_image.resize(new_size)  
  
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(resized_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  # create a binary (black and white) mask of the profile foreground
  mask = output_predictions.byte().cpu().numpy()
  background = np.zeros(mask.shape)
  bin_mask = np.where(mask, 255, background).astype(np.uint8)

  foreground = make_transparent_foreground(resized_image ,bin_mask)

  return foreground, bin_mask

deeplab_model = load_model()

@app.route("/")
def hello_world():
    return redirect('/upload-image')

@app.route("/upload-image")
def upload():    
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def uploadFile():
    uploaded_file = request.files['image']
    if uploaded_file.filename != '':        
        uploaded_file.save('static/' + 'uploaded_image.jpg')        
        session['imageForWork'] = 'uploaded_image.jpg'
        return redirect('/removebg')                         
    return 'No file selected'

@app.route("/removebg")
def removeBg():    
    imageForWork = session['imageForWork']            
    return render_template('removebg.html', image_upload = imageForWork)

@app.route('/remove', methods=['POST'])
def remove():
    imageForWork = session['imageForWork']
    foreground, bin_mask = remove_background(deeplab_model, 'static/'+imageForWork)    
    new_image = Image.fromarray(foreground)    
    new_image.save('static/result.png')
    return redirect('/success')

@app.route('/success')
def success():
    bgRemoved = 'result.png'
    return render_template('bg-removed.html', bgRemoved = bgRemoved)

if __name__ == '__main__':
    app.run(debug=True)
