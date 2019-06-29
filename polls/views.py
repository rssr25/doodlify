from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
	
	return render(request, "home/index.html")

def generateStyle(request):
	
	return HttpResponse('Generating Style')
