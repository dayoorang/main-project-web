import cv2
import matplotlib.pyplot as plt
import easyocr
import sys
import googletrans
from typing import List
import requests
import numpy as np
from PIL import Image, ImageFont, ImageDraw



CLIENT_ID = "MawiiHEojSbWlRvZjWEM"
CLIENT_SECRET = "gY1PNWHP54"


def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def easy_ocr_result(img, language='en', draw=True, text=False):
    reader = easyocr.Reader([language])
    print('img type',type(img))
    results = reader.readtext(img)
    print('results',results)
    # 바운딩박스 리스트
    bbox_list = []
    # 텍스트 리스트
    text_list = []

    for (bbox, text, prob) in results:
        # display the OCR'd text and associated probability
        # print("[INFO] {:.4f}: {}".format(prob, text))
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

            
        bbox_list.append((tl, tr, br, bl))
        text_list.append(text)

    return np.array(bbox_list), text_list


def translate_texts(texts: List[str], type='google') -> List[str]:
    global tranlated_texts

    # text_no = len(texts)    

    # if type == 'google':
    #     translator = googletrans.Translator()
    #     tranlated_texts = [
    #         translator.translate(text=text, src='en', dest='ko').text
    #         for text in texts
    #     ]
    if type == 'google':
        translator = googletrans.Translator()
        tranlated_texts = [
            translator.translate(text=text, src='en', dest='ko').text
            for text in texts
        ]
    elif type == 'naver':
        url = "https://openapi.naver.com/v1/papago/n2mt"
        header = {"X-Naver-Client-Id": CLIENT_ID, "X-Naver-Client-Secret": CLIENT_SECRET}
        tranlated_texts = []
        for text in texts:
            data = {'text': text, 'source': 'en', 'target': 'ko'}
            response = requests.post(url, headers=header, data=data)
            rescode = response.status_code
            if rescode == 200:
                t_data = response.json()
                tranlated_texts.append(t_data['message']['result']['translatedText'])
            else:
                print("Error Code:", rescode)

    return tranlated_texts

def cut_image(img, bbox):
    x_min = bbox[0, 0]
    x_max = bbox[1, 0]
    y_min = bbox[0, 1]
    y_max = bbox[2, 1]

    img = img[y_min:y_max, x_min:x_max]

    return img

def rgb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 배경이 밝은 부분이 한 부분이라도 있으면

    # 수정필요함 (귀퉁이 4개중 2개 이상이 흰색이면 이런식으로 )
    flat_list = list(mask.ravel())
    if flat_list.count(0) > len(flat_list):
        return 0

def mask_image(img2):
    # masking 작업
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    # 글씨 색이 밝든 어둡든 masking 씌워주기
    return_rgb = rgb(img2)
    if return_rgb == 0:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def change_original(img_np,masked_img, bbox):
    x_min = bbox[0, 0]
    x_max = bbox[1, 0]
    y_min = bbox[0, 1]
    y_max = bbox[2, 1]

    img_np[y_min:y_max, x_min:x_max] =  masked_img
    return img_np


def decsion_font_size( bbox_hi, text):
    font_size = 1
    title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
    _, hi = title_font.getsize(text)
    while hi < bbox_hi:
        title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
        font_size += 1
        _, hi = title_font.getsize(text)
    return font_size

def change_color(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray 영상으로 만들기
    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU) # 마스킹

    if len(img_binary[img_binary > 250]) > len(img_binary[img_binary < 250]): 
        img_binary = cv2.bitwise_not(img_binary)

    masked = cv2.bitwise_and(img, img, mask = img_binary)


    b, g, r = cv2.split(masked)
    b, g, r = int(np.mean(b[b > 0])), int(np.mean(g[g > 0])), int(np.mean(r[r > 0]))

    return b,g,r

def change_bg_color(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray 영상으로 만들기
    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU) # 마스킹
    bg_binary = cv2.bitwise_not(img_binary)

    if len(img_binary[img_binary > 250]) > len(img_binary[img_binary < 250]): 
        bg_binary = cv2.bitwise_not(bg_binary)

    masked_bg = cv2.bitwise_and(img, img, mask = bg_binary)

    b, g, r = cv2.split(masked_bg)
    b, g, r = int(np.mean(b[b > 0])), int(np.mean(g[g > 0])), int(np.mean(r[r > 0]))
    
    a = np.ones(shape=img.shape, dtype = np.uint8)
    b = a[:,:,0] * b
    g = a[:,:,1] * g
    r = a[:,:,2] * r

    
    return b,g,r

def rewrite(img, tranlated_texts ,bbox_list, color_list):

    img = img
    image_editable = ImageDraw.Draw(img)

    # (x, y ) , ( 237, 230, 211) 색감
    for idx, (bbox,color) in enumerate(zip(bbox_list,color_list)):      
        text = tranlated_texts[idx]
        title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', 1)
        wi, _ = title_font.getsize(text)
        # bbox_wi = bbox[1][0] - bbox[0][0]
        bbox_hi = bbox[2][1] - bbox[1][1]

        font_size = decsion_font_size(bbox_hi, text)
        title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
        image_editable.text((bbox[0][0], bbox[0][1]), text, color, anchor = 'lt', font=title_font)

    return img
    
    # print('img type',type(img))

    # save_rewrite_images(img)


