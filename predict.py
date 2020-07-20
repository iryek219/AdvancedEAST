import argparse

import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import cfg
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms

import os
import time
import datetime


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)

def draw_activation(y, im, activation_pixels, act_file):
    print("\nDrawing activation in ", act_file)
    draw = ImageDraw.Draw(im)
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        px = (j + 0.5) * cfg.pixel_size
        py = (i + 0.5) * cfg.pixel_size
        line_width, line_color = 1, 'red'
        if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
            if y[i, j, 2] < cfg.trunc_threshold:
                line_width, line_color = 2, 'yellow'
            elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                line_width, line_color = 2, 'green'
        draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                   (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                   (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                   (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                   (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                  width=line_width, fill=line_color)
    im.show()
    im.save(act_file)

def draw_prediction(im, pred):
    print("\nDrawing prediction")

def predict(east_detect, img_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)

    d_width, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_width, d_height), Image.NEAREST).convert('RGB')

    img = image.img_to_array(img)
    img = preprocess_input(img) 
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms, word_list = nms(y, activation_pixels)

    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_width, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_width / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_width, d_height), Image.NEAREST).convert('RGB')

        quad_im = im.copy()
        draw_activation(y,im,activation_pixels, img_path+'_act.jpg')
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='red')
                if cfg.predict_cut_text_line:
                    cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                  img_path, s)
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        
        for word in word_list:
            quad_draw.line([tuple(word[0], word[1], word[2], word[3], word[0])], width=2, fill='blue')
        quad_im.save(img_path + '_predict.jpg')
        if cfg.predict_write2txt and len(txt_items) > 0:
            with open(img_path[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)

def predict_word(east_detect, img, max_predict_img_size, pixel_threshold, 
                predict_cut_text_line, predict_write2txt, img_path, quiet=False):
    im_act = img.copy()
    im_act_array = image.img_to_array(im_act.convert('RGB'))

    d_width, d_height = resize_image(img, max_predict_img_size)
    img = img.resize((d_width, d_height), Image.NEAREST).convert('RGB')

    im_act = im_act.resize((d_width, d_height), Image.NEAREST).convert('RGB')
    scale_ratio_w = d_width / im_act.width
    scale_ratio_h = d_height / im_act.height
    quad_im = im_act.copy()

    img = image.img_to_array(img)
    img = preprocess_input(img) 
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)

    activation_pixels = np.where(cond)
    draw_activation(y, im_act, activation_pixels, img_path+'_act.jpg')

    #quad_scores, quad_after_nms = nms(y, activation_pixels)
    quad_scores, quad_after_nms, word_list = nms(y, activation_pixels)

    quad_draw= ImageDraw.Draw(quad_im)
    txt_items = []
    for score, geo, s in zip(quad_scores, quad_after_nms, range(len(quad_scores))):
            if np.amin(score) > 0:
                quad_draw.line([tuple(geo[0]), tuple(geo[1]), tuple(geo[2]), 
                                tuple(geo[3]), tuple(geo[0])], width=2, fill='red')
                if predict_cut_text_line:
                    cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_act_array,
                                  img_path, s)
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
    for word in word_list:
        quad_draw.line([tuple(word[0]), tuple(word[1]), tuple(word[2]), \
            tuple(word[3]), tuple(word[0])], width=2, fill='blue')
    quad_im.show()
    quad_im.save(img_path + '_predict.jpg')
    if predict_write2txt and len(txt_items) > 0:
        with open(img_path[:-4] + '.txt', 'w') as f_txt:
            f_txt.writelines(txt_items)

def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_width, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_width / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_width, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img) # Hwan - , mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/012.png',
                        help='image path')
    parser.add_argument('--dir', '-d',
                        default='',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    img_path = args.path
    img_dir = args.dir
    threshold = float(args.threshold)
    print(img_path, threshold)

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)

    if len(img_dir)==0:
        #predict(east_detect, img_path, threshold)
        with Image.open(img_path) as im:
            predict_word(east_detect, im, cfg.max_predict_img_size,
                        cfg.pixel_threshold,
                        cfg.predict_cut_text_line,
                        cfg.predict_write2txt, 
                        img_path)
    else:
        start_time = time.time()
        count = 0
        if not img_dir.endswith('/'):
            img_dir += '/'
        print("\n@@@@@ Folder: "+img_dir)
        for f in os.listdir(img_dir):
            if f.endswith('.png') or f.endswith('jpg'):
                print('\n',f)
                with Image.open(img_dir+f) as im:
                    predict_word(east_detect, im, cfg.max_predict_img_size,
                        cfg.pixel_threshold,
                        cfg.predict_cut_text_line,
                        cfg.predict_write2txt, 
                        img_dir)
                    count += 1
                    elapsed = time.time() - start_time
                    times = str(datetime.timedelta(seconds=elapsed)).split('.')
                    print('#{0:d} '.format(count), times[0], ' '+f)


'''
    start_time = time.time()
    count = 0

    dir_root = '../OCR-Test-Data/LOTT_E_REAL'
    img_dir = ['/A003/', '/ETC/']
    for p in img_dir:
        d = dir_root+p
        print("\n@@@@@ Folder: "+d)
        for f in os.listdir(d):
            if f.endswith('.png') or f.endswith('jpg'):
                print('\n',f)
                predict(east_detect, d+f, threshold)
                count += 1
                elapsed = time.time() - start_time
                times = str(datetime.timedelta(seconds=elapsed)).split('.')
                print('#{0:d} '.format(count), times[0], ' '+d+f)
'''