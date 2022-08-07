import cv2
from matplotlib.pyplot import axis
import numpy as np
import tflite_runtime.interpreter as tflite

def resize2(img, size, interpolation):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w:
        img = img/255.
        img = cv2.resize(img, (size, size), interpolation)
        return np.expand_dims(img.astype(np.float32), axis=0), None, None, None, None

    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w))
    y_pos = int((dif - h))
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    img = cv2.resize(mask, (size, size), interpolation)
    if h > w: tr = h/img.shape[0]
    else:     tr = w/img.shape[1]
    
    img = img/255.
    return np.expand_dims(img.astype(np.float32), axis=0), y_pos/tr, (y_pos+h)/tr, x_pos/tr, (x_pos+w)/tr
    

def run_style_predict(preprocessed_style_image, path):
    # Load the model.
    interpreter = tflite.Interpreter(model_path=path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
        )()

    return style_bottleneck

def run_style_transform(style_bottleneck, preprocessed_content_image, path):
    # Load the model.
    interpreter = tflite.Interpreter(model_path=path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
        )()

    return stylized_image

def superes(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "models/LapSRN_x4.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 4)
    result = sr.upsample(image)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

def blending(content, style, path1, path2, content_blending_ratio):
    content_style,_,_,_,_ = resize2(content, 256, cv2.INTER_AREA)
    style,_,_,_,_ = resize2(style, 256, cv2.INTER_AREA)
    content, y1, y2, x1, x2 = resize2(content, 384, cv2.INTER_AREA)

    style_bottleneck_content = run_style_predict(content_style, path1)
    style_bottleneck = run_style_predict(style, path1)
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
                           + (1 - content_blending_ratio) * style_bottleneck
    stylized_image_blended = run_style_transform(style_bottleneck_blended, content, path2)
    stylized_image_blended = np.squeeze(stylized_image_blended)

    if y1 == None:
        image = stylized_image_blended*255
        return superes(image)
    
    else:
        y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
        stylized_image_blended = stylized_image_blended[y1:y2, x1:x2]
        image = stylized_image_blended*255

        image = superes(image)

        return image

import numpy as np

def convert_single_example(text_a, max_seq_length,
                            tokenizer):

    tokens_a = tokenizer.tokenize(text_a)

    seg_id_a = 0
    seg_id_cls = 0
    seg_id_pad = 0

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(seg_id_cls)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(seg_id_a)
    tokens.append("[SEP]")
    segment_ids.append(seg_id_a)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(seg_id_pad)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return (np.array(input_ids), np.array(input_mask), np.array(segment_ids))

