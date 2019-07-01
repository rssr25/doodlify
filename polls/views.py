from django.shortcuts import render
from django.http import HttpResponse
import cv2

# Create your views here.

def index(request):
	
	return render(request, "home/index.html")

def generateStyle(request):
	
	contentImageFromPage = request.GET.get('contentImage')
	styleImageFromPage = request.GET.get('styleImage')
	contentImageLocation = "static/images/content_images/" + contentImageFromPage
	styleImageLocation = "static/images/style_images/" + styleImageFromPage

	contentImage = cv2.imread(contentImageLocation)
	styleImage = cv2.imread(styleImageLocation)

	contentImageGrayScale = cv2.cvtColor(contentImage, cv2.COLOR_BGR2GRAY)
	saveImageLocation = "static/images/generated_images/gray.jpg"

	cv2.imwrite(saveImageLocation, contentImageGrayScale)



	return HttpResponse(str(contentImage.shape) + " " + str(styleImage.shape))
	
