from django.forms import ModelForm
from django import forms
from inpainting.models import Image


class ImageCreationForm(ModelForm):
    image = forms.FileField(widget=forms.ClearableFileInput(attrs={"style" :"display:none",
                                                                   "name":"inpFile",
                                                                    "id":"inpFile" }))
    class Meta:
        model = Image
        fields = ['image']

