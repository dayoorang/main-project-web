from django.db import models
import matplotlib.pyplot as plt
import PIL
from io import BytesIO
from django.core.files.base import ContentFile
import numpy as np 
import os
import cv2
from pathlib import Path
from TEST_JW.test import (
    easy_ocr_result,
    translate_texts,
    cut_image,
    mask_image,
    change_original,
    change_color,
    rewrite,
    change_bg_color,
)

'''
변경된 이미지 출력 과정 :
---> 
1. Image 필드를 통해 image를 전달 받은 후, 
내부 알고리즘을 이미지를 수정하고 수정된 이미지를 DB상에 저장
2. 이미지를 웹페이지 출력할때는 DB에서 변경된 이미지를 불러오는식으로 진행
'''

class Image(models.Model):
    image = models.ImageField(upload_to='image/',blank=True, null=True)

    def save(self, *args, **kwargs): # save 모델을 수정하여 1번 과정을 수행

        img_pil = PIL.Image.open(self.image).convert('RGB') 
        # 우리가 사용할 easy-ocr모델에는 4채널 이미지가 들어올 경우 오류가 발생함으로 강제적으로 3채널 이미지로 제한


        img_np = np.array(img_pil)
        # 이미지 수정/변경 편의를 위해 넘파이 array 형태로 변경

        print('img_np shape', img_np.shape)
        bbox_list, text_list = easy_ocr_result(img_np)
        # 넘파이 array형태의 이미지를 easy_ocr_result 함수의 인자로 사용
        # 함수 내부에서 이미지안에 있는 text와, bounding box 위치를 리스트 형태로 각각 반환.

        
        tranlated_texts = translate_texts(texts=text_list, type='naver')
        # translate_texts에 위에서 return 받은 text_list를 전달받게 되는데, 이는 각 리스트들을 한글로 재번역하기 위함,
        # 두번째 인자로 type='naver'은 naver translator api를 사용하기 위한 코드이다.



        ## 각각의 text 객체들을 개별로 수정하기 위해 easy_ocr_result 함수에서 얻은 bounding box 리스트를 for loop을 통해 하나씩 뽑아낸다. ## 
        color_list=[]
        for bbox in bbox_list:
            img_cut = cut_image(img_np, bbox) # cut_image 함수의 첫번째 넘파이 이미지 array를, 두번째 인자로는 바운딩박스가 들어가게 된다.
            # for loop을 통해 얻은 각각의 바운딩 박스를 통해 전체 이미지에서 글자만 있는 부분을 cutting 하게되고 이를 통해 얻은 값은 img_cut 이라는 변수에 담았다.

            color_list.append(change_color(img_cut)) # cutting 된 이미지는 change_color 라는 함수에 인자로 넣는데, 이는 이미지의 포함되어있는 글자의 컬러를 뽑아내고,
            # 각 객체의 컬러들을 리스트에 담는 코드이다.

            print('img_cut shape',img_cut.shape)
            b,g,r = change_bg_color(img_cut) # 각 cutting된 이미지의 배경 컬러를 뽑아내는 코드이다.(자세한 설명은 test.py 코드 설명 참조)

            # 원본이미지에 수정된 이미지를 덮어 글자를 지운다.
            img_cut[:,:,0] = b
            img_cut[:,:,1] = g
            img_cut[:,:,2] = r


            img_np = change_original(img_np, img_cut, bbox)
            # change_original 함수의 첫번째 인자로는 원본이미지, 두번째 인자로는 수정된 커팅 이미지, 세번째 인자로는 바운딩박스가 들어간다.
            # 각각의 인자를 이용해 원본 이미지를 for loop 순으로 순차적으로 수정해 나간다.
            # 여기 까지 코드는 원본 이미지에 있는 글자를 덮는 기능이다.



        print('color list',color_list)

        img_pil = PIL.Image.fromarray(img_np)
        # 수정된 넘파이 array 이미지를 다시 Pillow 이미지 형태로 변경한다.(글자를 이미지에 적을 pillow 함수를 사용하기 위함)

        img = rewrite(img_pil, tranlated_texts,bbox_list, color_list)
        # rewrite 함수의 첫번째 인자로는 pillow형태의 이미지, 두번째 인자로는 번역된 글(한글) 리스트, 세번째 인자로는 바운딩박스 리스트, 세번째로는 각 글자의 컬러를 담은 리스트이다.
        # rewrite 를 통해 나온 최종 수정된 이미지는 배경이 지워지고, 원본 글자(영어)의 컬러를 따서 원래 위치에 넣어준 이미지이다.

        # 최종 수정된 이미지를 장고가 읽을 수있도록 변경 하는 코드
        buffer = BytesIO()
        img.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)

        super().save(*args,**kwargs)