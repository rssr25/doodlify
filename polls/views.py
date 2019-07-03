from django.shortcuts import render
from django.http import HttpResponse
from django.template import engines
import time
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
	saveImageLocation = "static/images/generated_images/" + contentImageFromPage + "_gray.jpg"

	cv2.imwrite(saveImageLocation, contentImageGrayScale)

	template_code = """<!doctype html><head><title>Your art image</title></head><body><img src = {{ saveImageLocation }}/> </body></html>"""

	template = engines['django'].from_string(template_code)

	time.sleep(5)

	return HttpResponse(template.render(context={'saveImageLocation': saveImageLocation}))
	
