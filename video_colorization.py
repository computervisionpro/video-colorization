
import cv2
import numpy as np

from openvino.runtime import Core


PRECISION = "FP16"
MODEL_NAME = "colorization-v2"
MODEL_PATH = f"{MODEL_NAME}/{PRECISION}/{MODEL_NAME}.xml"
DATA_DIR = "data"
DEVICE = "CPU"

# openvino inference
ie = Core()
model = ie.read_model(model=MODEL_PATH)
compiled_model = ie.compile_model(model=model, device_name=DEVICE)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
N, C, H, W = list(input_layer.shape)



def colorize(gray_img: np.ndarray):

    """
    Given an image as ndarray for inference convert the image into LAB image,
    the model consumes as input L-Channel of LAB image and provides output
    A & B - Channels of LAB image. i.e returns a colorized image

        Parameters:
            gray_img (ndarray): Numpy array representing the original
                                image.

        Returns:
            colorize_image (ndarray): Numpy arrray depicting the
                                      colorized version of the original
                                      image.
    """

    # Preprocess
    h_in, w_in, _ = gray_img.shape
    #print('shape gray', gray_img.shape)
    img_rgb = gray_img.astype(np.float32) / 255
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l = cv2.resize(img_lab.copy(), (W, H))[:, :, 0]
    #print('shape lab, l', img_lab.shape, img_l_rs.shape)

    # Inference
    inputs = np.expand_dims(img_l, axis=[0, 1])
    #print('shape inputs', inputs.shape)
    res = compiled_model([inputs])[output_layer]
    update_res = np.squeeze(res)

    # Post-process
    out = update_res.transpose((1, 2, 0))
    out = cv2.resize(out, (w_in, h_in))
    
    # adding a single channel
    img_lab = img_lab[:, :, 0][:, :, np.newaxis]
    img_lab_out = np.concatenate((img_lab, out), axis=2)

    # clip values beyond 0-1
    img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)
    colorized_image = (cv2.resize(img_bgr_out, (w_in, h_in))
                       * 255).astype(np.uint8)

    return colorized_image


cap = cv2.VideoCapture('./data/pexels-anna-tarazevich-5406325.mp4')
writer=None


fps = cap.get(cv2.CAP_PROP_FPS)
print('Frame rate of the video', fps)


print('Reading frames')
while cap.isOpened():

    
    ret, raw_image = cap.read()

    if not ret:
        break

    # if image has more than one channel
    if raw_image.shape[2] > 1:
        gray_img = cv2.cvtColor(
                     cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY),
                     cv2.COLOR_GRAY2RGB)
    else:
        gray_img = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)

    # color the frame
    print('Coloring the frame....')
    color_image = colorize(gray_img)
    

    if writer is None:

        print('Writing to a video file..')
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        (H0,W0) = color_image.shape[:2]
        writer = cv2.VideoWriter("./data/clr-video.mp4", fourcc, int(fps), (W0,H0), True)

    #save video
    writer.write(color_image)

    
    key = ord('q')
    if cv2.waitKey(1) == key:
        break
        

cap.release()
cv2.destroyAllWindows()

print("\n\nDone!")



