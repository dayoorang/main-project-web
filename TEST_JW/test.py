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

    if type == 'google': # 구글 transalotr 이용
        translator = googletrans.Translator()
        tranlated_texts = [
            translator.translate(text=text, src='en', dest='ko').text
            for text in texts
        ]
    elif type == 'naver': # 네이버 translator 이용
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
    x_min = bbox[0, 0]  # 좌상단 
    x_max = bbox[1, 0]  # 좌하단 
    y_min = bbox[0, 1]  # 우상단
    y_max = bbox[2, 1]  # 우하단

    img = img[y_min:y_max, x_min:x_max] # 바운딩 박스 좌표를 통해 바운딩박스 크기의 이미지를 뽑아내는 과정

    return img

def rgb(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 배경이 밝은 부분이 한 부분이라도 있으면

    # 수정필요함 (귀퉁이 4개중 2개 이상이 흰색이면 이런식으로 )
    flat_list = list(mask.ravel())
    if flat_list.count(0) > len(flat_list)/2:
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


def decsion_font_size(bbox_hi, text):
    font_size = 1
    title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
    _, hi = title_font.getsize(text)
    while hi < bbox_hi:
        title_font = ImageFont.truetype('ttf/NotoSansKR-Bold.otf', font_size)
        font_size += 1
        _, hi = title_font.getsize(text)
    return font_size

def change_color(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 이진 영상으로 만들기 전 grayscale 영상으로 변경
    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU) # otsu 기법을 통해 grascaled된 영상으로 검은색과 흰색으로 이루어진 이진영상으로 변경
    # return 된 ret는 OTSU기법을 통해 계산된 threshold 이고 , img_binary는 이진영상이다.
    if len(img_binary[img_binary > 250]) > len(img_binary[img_binary < 250]):
        # 우리가 실제로 필요한 이진 영상의 형태는 글자(객체)는 하얀색, 배경은 검은색이다.
        # 하지만 영상에 따라 글자가 더 어두울 경우 글자를 검은색으로 분류를 하는 경우가 발생한다.
        # 이를 해결하기위해 임의의 숫자 250(하얀색에 가까운 수)를 기준으로 위와 같은 if문을 지정하였고 if문에 해당될 경우 반전하는 코드인 bitwise_not을 사용하였다. 
        # len(img_binary[img_binary > 250]) ==> 250보다 큰 픽셀값이 이미지에 몇개있는지를 체크 즉 하얀색 픽셀 갯수 체크
        # len(img_binary[img_binary < 250]) ==> 250보다 작은 픽셀값 체크 즉 검은색 픽셀 갯수 체크
        # 이 두값중 "len(img_binary[img_binary > 250])"이 더크다는 것은 배경이 하얀색이라는 의미임으로 반전을 해야한다.(전제: 객체보다 배경의 픽셀수가 더 많다 가정)
        img_binary = cv2.bitwise_not(img_binary)

    masked = cv2.bitwise_and(img, img, mask = img_binary) # 객체부분만 가져오기위한 코드


    b, g, r = cv2.split(masked) # 객체를 split함수를 통해 b, g, r 각각의 컬러로 분리
    b, g, r = int(np.mean(b[b > 0])), int(np.mean(g[g > 0])), int(np.mean(r[r > 0])) 
    # 글자(객체)의 컬러를 보면 solid 하기보단 그라데이션이 적용되기도하고 불투명한 경우도 많다.
    # 이러한 컬러 특징을 제대로 잡아내기는 현실적으로 어려움이 있어 컬러의 평균을 구해 b, g, r 로 재지정

    return b,g,r


def change_bg_color(img):
    # 위 change_color와 같은 방법으로 역을 구해 background 컬러를 구하는 과정
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU) 
    bg_binary = cv2.bitwise_not(img_binary)

    if len(img_binary[img_binary > 250]) > len(img_binary[img_binary < 250]): 
        bg_binary = cv2.bitwise_not(bg_binary)

    masked_bg = cv2.bitwise_and(img, img, mask = bg_binary)

    b, g, r = cv2.split(masked_bg)
    b, g, r = int(np.mean(b[b > 0])), int(np.mean(g[g > 0])), int(np.mean(r[r > 0]))
    

    a = np.ones(shape=img.shape, dtype = np.uint8)  # change_color와는 다르게 boundingbox 전체를 하나의 컬러로 덮는 용도이기 때문에 컷팅된 이미지의 shape크기만큼의 이미지를 생성
    b = a[:,:,0] * b # 각각의 컬러 만큼 곱해주고 b, g, r에 담는다.
    g = a[:,:,1] * g
    r = a[:,:,2] * r

    
    return b,g,r

'''
rewrite 함수는 원본 이미지에서 원래 글자가 있는 부분을 지운 형태의 이미지를 첫번째 인자로 전달 받음,
해당함수의 목적은 원본 이미지에 번역 된 글자를 덮어 쓰기 위함
'''
def rewrite(img, tranlated_texts ,bbox_list, color_list):
    img = img
    image_editable = ImageDraw.Draw(img)

    ### font size ###
    '''' 
    원본이미지의 폰트사이즈를 이용하여 번역된 글자 폰트 사이즈를 정하는 코드
    원본이미지의 폰트사이즈는 해당 글자(객체) 바운딩 박스 높이를 기준으로 하였으나, 같은 크기의 글자임에도 바운딩박스 크기에 따라 
    글자 크기가 변하는 문제 발생

    이 문제를 해결 하기 위해 각 바운딩박스의 median값을 구하고 median값과 차이가 큰 값들은 특별히 작거나 큰 값이므로, 원래 값을 그대로 이용하고
    차이가 작은 값들은 모두 같은 크기(여기선 median값)으로 변경함으로써 문제 해결
    '''
    bbox_hi = []
    for bbox in  bbox_list:
        bbox_hi.append(bbox[2][1] - bbox[0][1]) 
    # 위 문제를 해결하기 위해 각 바운딩박스의 높를 리스트에 담음


    bbox_hi_median = int(np.median(bbox_hi)) # 리스트에 담은 값들의 중앙값 추출
    bbox_hi_median_diff = np.abs(np.array(bbox_hi) - bbox_hi_median ) # 각각의 값들이 중앙 값에서 얼마나 벗어나있는지를 알기 위한 코드
    print('bbox_hi_median',bbox_hi_median)
    hi_lt_idx = np.where(bbox_hi_median_diff < 10) # 10보다 적게나는 값 idx 추출(10은 테스트를 통해 가장 좋은 값으로 정함)

    # bbox_hi > array 변경
    bbox_hi = np.array(bbox_hi) # 계산 편의를 위해 넘파이로 변경
    bbox_hi[hi_lt_idx] = bbox_hi_median   # 차이가 작은 값은 median값으로 변경하고 큰것은 원래 값 그대로 사용.


    print('bbox_hi', bbox_hi)
    for idx, (bbox,color) in enumerate(zip(bbox_list,color_list)):  
          
        print('fontsize',bbox_hi[idx]-15)
        text = tranlated_texts[idx]

        title_font = ImageFont.truetype("ttf/NotoSansKR-Bold.otf", np.maximum(2, bbox_hi[idx]-5)) 
        # ImageFont.truetype 두번째 인자가 폰트 사이즈가 들어가는 위치인데, -5 정도를 빼줘야 크기가 비슷함
        # 다만 크기가 작은 텍스트일 경우 -가 되버리는 경우가 있기 때문에 최소 크기 2가 되도록 고정
        wi, hi = title_font.getsize(text)

        print('bbox_hi[idx]-15', type(bbox_hi[idx]-15))
        
        print('title_font',title_font)
        start_x = ((bbox[0][0] + bbox[1][0]) // 2  -wi / 2)
        start_y = ((bbox[0][1] + bbox[2][1]) // 2 - hi / 2)
        image_editable.text((start_x, start_y), text, color, anchor = 'lt', font=title_font)

    return img



