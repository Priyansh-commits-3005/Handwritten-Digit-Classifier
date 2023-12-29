''' this is basically to make a somewhat of a frontend to the model made'''


import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
import torchvision
import torch



#the basic title and the working 

st.title(" :red[Handwritten] Digit Predictor")
st.header(" Using a Sequential :orange[Tensorflow.Keras] Model",divider= 'blue')
new_model = tf.keras.models.load_model("C:\codes\Projects\project detection\handwritten digit recognizer\Handwritten-Digit-Classifier\model\digit-recognizer.h5")
st.write("#### Draw a digit from 0-9 in the below box for the model to predict the digit")

# Specify canvas parameters in application

strokeWidth = st.sidebar.slider("Stroke Width : ",1,25,9)
realtime_update = st.sidebar.checkbox("Turn realitime updating on or off",True)

# the drawable canvas here

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=strokeWidth,
    stroke_color="#FFFFFF",
    background_color="#000000",
    update_streamlit = realtime_update,
    height = 200,
    width = 200,
    drawing_mode = 'freedraw',
    key = 'canvas'
)




if canvas_result.image_data is not None:
    input_numpy_array = np.array(canvas_result.image_data)
     
     
    # Get the RGBA PIL image
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image.save('user_input.png')
     
    # Convert it to grayscale
    input_image_gs = input_image.convert('L')
    input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200,200)
    # st.write('### Image as a grayscale Numpy array')
    # st.write(input_image_gs_np)
     
    # Create a temporary image for opencv to read it
    input_image_gs.save('temp_for_cv2.jpg')
    image = cv2.imread('temp_for_cv2.jpg', 0)
    # Start creating a bounding box
    height, width = image.shape
    x,y,w,h = cv2.boundingRect(image)
 
 
    # Create new blank image and shift ROI to new coordinates
    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
    width, height = mask.shape
#     print(ROI.shape)
#     print(mask.shape)
    x = width//2 - ROI.shape[0]//2
    y = height//2 - ROI.shape[1]//2
#     print(x,y)
    mask[y:y+h, x:x+w] = ROI
#     print(mask)
    # Check if centering/masking was successful
#     plt.imshow(mask, cmap='viridis') 
    output_image = Image.fromarray(mask) # mask has values in [0-255] as expected
    # Now we need to resize, but it causes problems with default arguments as it changes the range of pixel values to be negative or positive
    # compressed_output_image = output_image.resize((22,22))
    # Therefore, we use the following:
    compressed_output_image = output_image.resize((22,22), Image.BILINEAR) # PIL.Image.NEAREST or PIL.Image.BILINEAR also performs good
 
    convert_tensor = torchvision.transforms.ToTensor()
    tensor_image = convert_tensor(compressed_output_image)
    # Another problem we face is that in the above ToTensor() command, we should have gotten a normalized tensor with pixel values in [0,1]
    # But somehow it doesn't happen. Therefore, we need to normalize manually
    tensor_image = tensor_image/255.
    # Padding
    tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
    # Normalization shoudl be done after padding i guess
    convert_tensor = torchvision.transforms.Normalize((0.1307), (0.3081)) # Mean and std of MNIST
    tensor_image = convert_tensor(tensor_image)
    # st.write(tensor_image.shape) 
    # Shape of tensor image is (1,28,28)
     
 
 
    # st.write('### Processing steps:')
    # st.write('1. Find the bounding box of the digit blob and use that.')
    # st.write('2. Convert it to size 22x22.')
    # st.write('3. Pad the image with 3 pixels on all the sides to get a 28x28 image.')
    # st.write('4. Normalize the image to have pixel values between 0 and 1.')
    # st.write('5. Standardize the image using the mean and standard deviation of the MNIST_plus dataset.')
 
    # The following gives noisy image because the values are from -1 to 1, which is not a proper image format
    im = Image.fromarray(tensor_image.detach().cpu().numpy().reshape(28,28), mode='L')
    im.save("processed_tensor.png", "PNG")
    # So we use matplotlib to save it instead
    plt.imsave('processed_tensor.png',tensor_image.detach().cpu().numpy().reshape(28,28), cmap='gray')
 
    # st.write('### Processed image')
    # st.image('processed_tensor.png')
    # st.write(tensor_image.detach().cpu().numpy().reshape(28,28))
    ## conversion to tf tensor
    np_tensor = tensor_image.numpy()
    tensor_image = tf.convert_to_tensor(np_tensor)
    ##prediction
    prediction = new_model.predict(tensor_image)  
    st.write(np.argmax(prediction))