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

	
def build_model(input_img):
  if args.verbose: print('\nBUILDING VGG-19 NETWORK')
  net = {}
  _, h, w, d     = input_img.shape
  
  if args.verbose: print('loading model weights...')
  vgg_rawnet     = scipy.io.loadmat(args.model_weights)
  vgg_layers     = vgg_rawnet['layers'][0]
  if args.verbose: print('constructing layers...')
  net['input']   = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

  if args.verbose: print('LAYER GROUP 1')
  net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
  net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

  net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
  net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))
  
  net['pool1']   = pool_layer('pool1', net['relu1_2'])

  if args.verbose: print('LAYER GROUP 2')  
  net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
  net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))
  
  net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
  net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))
  
  net['pool2']   = pool_layer('pool2', net['relu2_2'])
  
  if args.verbose: print('LAYER GROUP 3')
  net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
  net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

  net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
  net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

  net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
  net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

  net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
  net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

  net['pool3']   = pool_layer('pool3', net['relu3_4'])

  if args.verbose: print('LAYER GROUP 4')
  net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
  net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

  net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
  net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

  net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
  net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

  net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
  net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

  net['pool4']   = pool_layer('pool4', net['relu4_4'])

  if args.verbose: print('LAYER GROUP 5')
  net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
  net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

  net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
  net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

  net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
  net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

  net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
  net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

  net['pool5']   = pool_layer('pool5', net['relu5_4'])

  return net

def conv_layer(layer_name, layer_input, W):
  conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
  if args.verbose: print('--{} | shape={} | weights_shape={}'.format(layer_name, 
    conv.get_shape(), W.get_shape()))
  return conv

def relu_layer(layer_name, layer_input, b):
  relu = tf.nn.relu(layer_input + b)
  if args.verbose: 
    print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.get_shape(), 
      b.get_shape()))
  return relu

def pool_layer(layer_name, layer_input):
  if args.pooling_type == 'avg':
    pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], 
      strides=[1, 2, 2, 1], padding='SAME')
  elif args.pooling_type == 'max':
    pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], 
      strides=[1, 2, 2, 1], padding='SAME')
  if args.verbose: 
    print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
  return pool

def get_weights(vgg_layers, i):
  weights = vgg_layers[i][0][0][2][0][0]
  W = tf.constant(weights)
  return W

def get_bias(vgg_layers, i):
  bias = vgg_layers[i][0][0][2][0][1]
  b = tf.constant(np.reshape(bias, (bias.size)))
  return b

'''
  Loss Functions
'''
def content_layer_loss(p, x):
  _, h, w, d = p.get_shape()
  M = h.value * w.value
  N = d.value
  if args.content_loss_function   == 1:
    K = 1. / (2. * N**0.5 * M**0.5)
  elif args.content_loss_function == 2:
    K = 1. / (N * M)
  elif args.content_loss_function == 3:  
    K = 1. / 2.
  loss = K * tf.reduce_sum(tf.pow((x - p), 2))
  return loss

def style_layer_loss(a, x):
  _, h, w, d = a.get_shape()
  M = h.value * w.value
  N = d.value
  A = gram_matrix(a, M, N)
  G = gram_matrix(x, M, N)
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
  return loss

def gram_matrix(x, area, depth):
  F = tf.reshape(x, (area, depth))
  G = tf.matmul(tf.transpose(F), F)
  return G

def mask_style_layer(a, x, mask_img):
  _, h, w, d = a.get_shape()
  mask = get_mask_image(mask_img, w.value, h.value)
  mask = tf.convert_to_tensor(mask)
  tensors = []
  for _ in range(d.value): 
    tensors.append(mask)
  mask = tf.stack(tensors, axis=2)
  mask = tf.stack(mask, axis=0)
  mask = tf.expand_dims(mask, 0)
  a = tf.multiply(a, mask)
  x = tf.multiply(x, mask)
  return a, x

def sum_masked_style_losses(sess, net, style_imgs):
  total_style_loss = 0.
  weights = args.style_imgs_weights
  masks = args.style_mask_imgs
  for img, img_weight, img_mask in zip(style_imgs, weights, masks):
    sess.run(net['input'].assign(img))
    style_loss = 0.
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
      a = sess.run(net[layer])
      x = net[layer]
      a = tf.convert_to_tensor(a)
      a, x = mask_style_layer(a, x, img_mask)
      style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(len(args.style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_style_losses(sess, net, style_imgs):
  total_style_loss = 0.
  weights = args.style_imgs_weights
  for img, img_weight in zip(style_imgs, weights):
    sess.run(net['input'].assign(img))
    style_loss = 0.
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
      a = sess.run(net[layer])
      x = net[layer]
      a = tf.convert_to_tensor(a)
      style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(len(args.style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_content_losses(sess, net, content_img):
  sess.run(net['input'].assign(content_img))
  content_loss = 0.
  for layer, weight in zip(args.content_layers, args.content_layer_weights):
    p = sess.run(net[layer])
    x = net[layer]
    p = tf.convert_to_tensor(p)
    content_loss += content_layer_loss(p, x) * weight
  content_loss /= float(len(args.content_layers))
  return content_loss