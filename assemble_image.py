import json
import ast
from PIL import Image, ImageDraw, ImageFont
import math
import PIL.ExifTags as ExifTags
from itertools import groupby
from operator import itemgetter
from functools import reduce
import numpy as np
import random
from multiprocessing import Pool, cpu_count, Manager
from multiprocessing import sharedctypes
import os.path
import sys
from PIL import ImageFilter
import itertools
from optparse import OptionParser
import shutil
import textwrap
parser = OptionParser()
parser.add_option("-o", "--overlap", dest="overlap", action = "store_true", default = False,
                  help="if images can overlap, if false, then images will cut to equal squares")
parser.add_option("-c", "--cut", dest="cut", default = 10, type = int,
                  help="the pixel square size you want for the output image")
parser.add_option("-e", "--enlarge", dest="enlarge", default = 10, type = int,
                  help="how much you want to enlarge each square")
parser.add_option("-m", "--mask", dest="mask", default = 0.25, type = float,
                  help="a value that determines what percentage of the original value will mask the output image, min 0 max 1")
parser.add_option("-i", "--image", dest = "input_image", type = "string",
				  help = "the image that you want to assemble")



## those options are for assembling text instead of an image
parser.add_option("-t", "--text", dest="assemble_text", action = "store", type = "string", 
				  help = "instead of assembling an input image, this option will allow you to assemble a string")
parser.add_option("--top", dest="top_margin", action = "store", type = "float", default = 0.2,
				  help = "stores what percent the top margin will occupy, range from 0 to 1")
parser.add_option("--bot", dest="bot_margin", action = "store", type = "float", default = 0.2,
				  help = "stores what percent the bottom margin will occupy, range from 0 to 1")
parser.add_option("--right", dest="right_margin", action = "store", type = "float", default = 0.2,
				  help = "stores what percent the right margin will occupy, range from 0 to 1")
parser.add_option("--left", dest="left_margin", action = "store", type = "float", default = 0.2,
				  help = "stores what percent the left margin will occupy, range from 0 to 1")
parser.add_option("--font", dest = "font", action = "store", type = "string", default = "ariblk.ttf",
				  help = "the type of font that you want to use for the text")
parser.add_option("--wrap", dest = "wrap", action = "store", type = int, default = False,
				  help = "an int to define the wrap value when using text wrap")
parser.add_option("--square", dest = "square", action = "store", type = int, default = 5,
				  help = "how large each square should be (in pixel size)")


(options, args) = parser.parse_args()


print(options)
print(args)

random.seed(926)

square_size = int(options.cut)
enlarge_size = int(options.enlarge)
if not options.assemble_text:
	image_file = str(options.input_image)

	shade_value = float(options.mask)
else:
	msg = str(options.assemble_text)
	top_margin = options.top_margin
	bot_margin = options.bot_margin
	left_margin = options.left_margin
	right_margin = options.right_margin
	font = options.font
	wrap = options.wrap
	square = options.square
	W, H = (1, 1)
	fnt = ImageFont.truetype(font, 50)
	text_im = Image.new("RGBA",(W,H),"white")
	draw = ImageDraw.Draw(text_im)
	if not wrap:
		## we will create a 1 by 1 pixel picture first to determine how large the text size will be

		w, h = draw.textsize(msg, font = fnt)

		## reset the width and height of the text image
		W, H = (int(w/(1-left_margin - right_margin)), int(h/(1-top_margin-bot_margin)))

		text_im = Image.new("RGB",(W,H),"white")
		draw = ImageDraw.Draw(text_im)
		draw.text((int(left_margin*W), int(top_margin*H)), msg, fill="black", font = fnt)
		text_im = text_im.resize((W*2, H*2), Image.EXTENT)
		# text_im.show()
	else:
		lines = textwrap.wrap(msg, width=wrap)
		y_text = 0
		max_width = 0
		for line in lines:
		    width, height = fnt.getsize(line)
		    draw.text((0, y_text), line, font=fnt)
		    if width>max_width:
		    	max_width = width
		    y_text += height
		W, H = (int(max_width/(1-left_margin - right_margin)), int(y_text/(1-top_margin-bot_margin)))
		text_im = Image.new("RGB",(W,H),"white")
		draw = ImageDraw.Draw(text_im)

		y_text = int(round(top_margin*H))
		for line in lines:
			w, h = fnt.getsize(line)
			draw.text((int((W-w)/2), y_text), line, fill="black", font = fnt)
			y_text+=h
		text_im = text_im.resize((W*2, H*2), Image.EXTENT)







# test_output_img = "thumb.jpg"


print("Getting stored image color information")
current_folder = os.getcwd()
thumbnail_dictionary_str = json.load(open(current_folder+ os.sep+"data"+os.sep+ args[0]+"_data.json", 'r'))
# print(thumbnail_dictionary)

thumbnail_dictionary = {}
brightness_dict = {}

## make temp folder to store the thumbnail images
if not os.path.exists(os.getcwd()+os.sep+args[0]+"_thumbnail_images"):
    os.makedirs(os.getcwd()+os.sep+args[0]+"_thumbnail_images")


## store the image informations, the key is the rgb value and the value is the path to the image
for key in thumbnail_dictionary_str.keys():
	# print(key)
	tuple_key = ast.literal_eval(key)
	thumbnail_dictionary[tuple_key] = thumbnail_dictionary_str[key]
	if options.assemble_text:
		brightness = int(round((tuple_key[0]*299+tuple_key[1]*587+tuple_key[2]*114)/1000))
		if brightness in brightness_dict:
			brightness_dict[brightness]+= thumbnail_dictionary_str[key]
		else:
			brightness_dict[brightness] = thumbnail_dictionary_str[key]

print("Information process done")



def find_closest_rgb(pixel_value):
	# a method to find a set of close rgb values, the reason it does not return the closest one
	# is so that we can try to avoid using one image over and over again
	return_closest_rgb_list = []
	all_values = [(((x[0]-pixel_value[0])**2+ (x[1]-pixel_value[1])**2+ (x[2]-pixel_value[2])**2), x) for x in thumbnail_dictionary.keys()]
	all_values_sorted = sorted(all_values, key = lambda x: x[0])

	return_closest_rgb_list = [all_values_sorted[0][1]]
	for x in range(1, len(all_values_sorted)):
		if all_values_sorted[x][0] - all_values_sorted[0][0]<70:
			return_closest_rgb_list.append(all_values_sorted[x][1])
		else:
			break
	return return_closest_rgb_list


def find_brightness(brightness_value):
	close_list = np.random.normal(0, abs(brightness_value-128)/4, 8)
	close_list = [int(round(x)) for x in close_list]
	# print(close_list)

	if brightness_value>=128:
		close_list = [abs(int(256-abs(brightness_value+x-256))-128)+128 for x in close_list]

	else:
		close_list = [128-abs(int(abs(brightness_value+x))-128) for x in close_list]

	close_list = [int(round(0.3*x+0.7*brightness_value)) for x in close_list]
	close_list = [min(brightness_dict.keys(), key = lambda y: abs(y-x)) for x in close_list]
	return close_list

# print(find_brightness(248))








# brightness_list = []
# for item in brightness_dict.keys():
# 	brightness_list+=[item]*len(brightness_dict[item])

if options.assemble_text:
	dark = 0
	bright = 0
	for item in brightness_dict.keys():
		if item<=128:
			dark+=len(brightness_dict[item])
		else:
			bright+=len(brightness_dict[item])

	print(dark, bright)

	bright_dominant = bright>dark


def text_image_to_brightnessblock(text_img):
	width, height = text_img.size
	width = width - width%square
	height = height - height%square

	text_img = text_img.crop((0, 0, width, height))
	# text_img = text_img.filter(ImageFilter.GaussianBlur(radius = 3))
	brightness_block = []
	pixels = np.array(text_img)

	for i in range(height//square):
		tmp = []
		for j in range(width//square):
			if bright_dominant:
				tmp.append(int(round(np.sum(pixels[i*square:i*square+square, j*square:j*square+square, 0:3])/(3*square*square))))
			else:
				tmp.append(255-int(round(np.sum(pixels[i*square:i*square+square, j*square:j*square+square, 0:3])/(3*square*square))))
		brightness_block.append(tmp)
	return brightness_block






# plt.show()

def image_to_colorblock(image_file, square_size):
	# a method to look at the image and transfer each little square of image into a color block

	im = Image.open(image_file)
	## filter later because if we filter right now, the orientation information will be lost
	# im = im.filter(ImageFilter.GaussianBlur(radius=2))
	orient = -1
	try:
		exif=dict((ExifTags.TAGS[k], v) for k, v in im._getexif().items() if k in ExifTags.TAGS)
		orient = exif["Orientation"]
	except:
		pass



	if orient == 3:
		im = im.rotate(180, expand = True)
	elif orient == 6:
		im = im.rotate(270, expand = True)
	elif orient == 8:
		im = im.rotate(90, expand = True)

	width, height = im.size

	width = width - width%square_size
	height = height - height%square_size

	im = im.crop((0, 0, width, height))
	im = im.filter(ImageFilter.GaussianBlur(radius=2))
	# im.show()


	pixels = np.array(im)
	color_blocks = []

	# new_image = np.array(Image.new("RGB", (width, height)))
	# print(new_image.shape)

	# counter = 0
	for i in range(int(pixels.shape[0]/square_size)):
		tmp = []
		for j in range(int(pixels.shape[1]/square_size)):
			

			# print(i, j)
			r_average = np.sum(pixels[i*square_size:i*square_size+square_size, j*square_size:j*square_size+square_size, 0])//(square_size*square_size)
			g_average = np.sum(pixels[i*square_size:i*square_size+square_size, j*square_size:j*square_size+square_size, 1])//(square_size*square_size)
			b_average = np.sum(pixels[i*square_size:i*square_size+square_size, j*square_size:j*square_size+square_size, 2])//(square_size*square_size)
			tmp.append((r_average, g_average, b_average, pixels[i*square_size:i*square_size+square_size, j*square_size:j*square_size+square_size, :]))

		color_blocks.append(tmp)


	return color_blocks

# image_to_colorblock("test_image.jpg", 10)




def post_process(image_file):
	# cut the image into a square

	im = Image.open(image_file)
	orient = -1
	try:
		exif=dict((ExifTags.TAGS[k], v) for k, v in im._getexif().items() if k in ExifTags.TAGS)
		orient = exif["Orientation"]
	except:
		pass
	if orient == 3:
		im = im.rotate(180, expand = True)
	elif orient == 6:
		im = im.rotate(270, expand = True)
	elif orient == 8:
		im = im.rotate(90, expand = True)


	size = width, height = im.size

	if not options.overlap:
		if width>height:
			# print("something")
			im = im.crop(((width-height)/2, 0, width - (width-height)/2, height))
		elif height>width:
			im = im.crop((0, (height-width)/2, width, height - (height-width)/2))
		else:
			pass

		im = im.resize((square_size*enlarge_size, square_size*enlarge_size), Image.LANCZOS)
	else:
		if width>height:
			im = im.resize((int((width/height)*square_size*enlarge_size), square_size*enlarge_size), Image.LANCZOS)
		else:
			im = im.resize((square_size*enlarge_size, int((height/width)*square_size*enlarge_size)), Image.LANCZOS)
	return im


## store the colorblocks 
if not options.assemble_text:
	color_blocks = image_to_colorblock(image_file, square_size)
else:
	brightness_blocks = text_image_to_brightnessblock(text_im)
	# print(brightness_blocks)


def choose_image_for_text(pos, block_image):
	print(pos)
	brightness_value = brightness_blocks[pos[0]][pos[1]]
	print(str(pos)+ " brightness value is: "+str(brightness_value))
	closest_brightnest_value = find_brightness(brightness_value)
	possible_images = []
	for x in closest_brightnest_value:
		possible_images+=brightness_dict[x]
	choose_image = random.choice(possible_images)
	image_name = choose_image.split(os.sep)[-1]
	save_thumbnail_image_path = os.getcwd()+os.sep+args[0]+"_thumbnail_images"+os.sep+image_name.split(".")[0]+"_thumb_"+str(square_size*enlarge_size)+ str(options.overlap) +"."+image_name.split(".")[-1]
	try:
		img = Image.open(save_thumbnail_image_path)
		# block_image[pos] = save_thumbnail_image_path
	except:
		img = post_process(choose_image)
		img.save(save_thumbnail_image_path)
		# block_image[pos] = save_thumbnail_i
	equalized_image = os.getcwd()+os.sep+args[0]+"_thumbnail_images"+os.sep+image_name.split(".")[0]+"_thumb_"+str(square_size*enlarge_size)+"_"+str(pos)+"."+image_name.split(".")[-1]
	while (not os.path.isfile(equalized_image)):
		try:	
			# scipy.misc.toimage(img, cmin=0.0, cmax=255).save(equalized_image)
			img.save(equalized_image)
		except:
			pass
	block_image[pos] = equalized_image
	print(equalized_image)




def choose_image(pos, block_image):
	# choose image for a colorblock
	print(pos)
	color_value = color_blocks[pos[0]][pos[1]][0:3]


	print(str(pos)+" rgb value is: "+str(color_value))

	# this gives a list of very close rgb values
	closest_rgb = find_closest_rgb(color_value)
	print(str(pos)+ " closest rgb values stored is: "+ str(closest_rgb))

	# get the nine neighbor of the corrent position
	pos_neighbors = [(pos[0]+i, pos[1]+j) for i, j in list(itertools.product([-1, 0, 1], [-1, 0, 1]))]

	# check the images that are in the neighbor of current position
	already_existed_pics = list(filter(lambda x: x, [block_image[x] if x in block_image else None for x in pos_neighbors]))

	# get a list of pictures that correspond to the close rgb values
	possible_pictures = []

	for item in closest_rgb:
		possible_pictures+=thumbnail_dictionary[item]

	possible_pictures_copy = [x for x in possible_pictures]

	for item in already_existed_pics:
		if item in possible_pictures:
			possible_pictures.remove(item)

	if possible_pictures == []:
		choose_image = random.choice(possible_pictures_copy)
	else:
		choose_image = random.choice(possible_pictures)

	# choose_image = random.choice(thumbnail_dictionary[closest_rgb])


	image_name = choose_image.split(os.sep)[-1]
	save_thumbnail_image_path = os.getcwd()+os.sep+args[0]+"_thumbnail_images"+os.sep+image_name.split(".")[0]+"_thumb_"+str(square_size*enlarge_size)+ str(options.overlap) +"."+image_name.split(".")[-1]

	try:
		img = Image.open(save_thumbnail_image_path)
		# block_image[pos] = save_thumbnail_image_path
	except:
		img = post_process(choose_image)
		img.save(save_thumbnail_image_path)
		# block_image[pos] = save_thumbnail_image_path

	# get the mask
	if not options.overlap:
		origin_fragmant = np.array(Image.fromarray(color_blocks[pos[0]][pos[1]][3]).resize((square_size*enlarge_size, square_size*enlarge_size), Image.LANCZOS).convert('RGB'))
	else:
		origin_fragmant = np.ones((img.size[1], img.size[0], 3))
		origin_fragmant[:, :, 0:3] = color_value
	# origin_fragmant = origin_fragmant//4
	img = np.array(img.convert('RGB'))
	img = np.uint8((1-shade_value)*img+ shade_value*origin_fragmant)
	img = Image.fromarray(img)
	equalized_image = os.getcwd()+os.sep+args[0]+"_thumbnail_images"+os.sep+image_name.split(".")[0]+"_thumb_"+str(square_size*enlarge_size)+"_"+str(pos)+"."+image_name.split(".")[-1]
	while (not os.path.isfile(equalized_image)):
		try:	
			# scipy.misc.toimage(img, cmin=0.0, cmax=255).save(equalized_image)
			img.save(equalized_image)
		except:
			pass
	block_image[pos] = equalized_image
	print(equalized_image)
	



# start multiprocessing to choose image for each color block
if __name__ == "__main__":
	
	manager = Manager()
	block_image = manager.dict()

	## start the multi processing


	# list comprehension to get all the positions
	

	if not options.assemble_text:
		positions = list(itertools.product(list(range(len(color_blocks))), list(range(len(color_blocks[0])))))
		pool = Pool(cpu_count())
		print("Start multiprocessing to choose image")
		pool.starmap(choose_image, [(x, block_image) for x in positions])
		print("done processing")


		## make a new image for pasting
		new_image = Image.new("RGB", (len(color_blocks[0])*square_size*enlarge_size, len(color_blocks)*square_size*enlarge_size))


		## paste an image to the location
		def paste_image(item):
			x_pos = item[1]
			y_pos = item[0]
			print(item)
			im = Image.open(block_image[item])
			new_image.paste(im, (x_pos*square_size*enlarge_size, y_pos*square_size*enlarge_size))


		## shuffle the keys (it is useful if overlap is set to true)
		block_image_keys = block_image.keys()
		random.shuffle(block_image_keys)
		for item in block_image_keys:
			paste_image(item)


		## print the image name and save the image
		print("".join(options.input_image.split(".")[:-1])+"_out.jpg")
		new_image.save("".join(options.input_image.split(".")[:-1])+"_out.jpg")

		print("Image saved, clearing thumbnail images")


		## destroy the thumbnail image directory
		shutil.rmtree(os.getcwd()+os.sep+args[0]+"_thumbnail_images")


	else:
		positions = list(itertools.product(list(range(len(brightness_blocks))), list(range(len(brightness_blocks[0])))))
		pool = Pool(cpu_count())
		print("Start multiprocessing to choose image for text")
		pool.starmap(choose_image_for_text, [(x, block_image) for x in positions])
		print("done processing")


		## make a new image for pasting
		new_image = Image.new("RGB", (len(brightness_blocks[0])*square_size*enlarge_size, len(brightness_blocks)*square_size*enlarge_size))


		## paste an image to the location
		def paste_image(item):
			x_pos = item[1]
			y_pos = item[0]
			print(item)
			im = Image.open(block_image[item])
			new_image.paste(im, (x_pos*square_size*enlarge_size, y_pos*square_size*enlarge_size))


		## shuffle the keys (it is useful if overlap is set to true)
		block_image_keys = block_image.keys()
		random.shuffle(block_image_keys)
		for item in block_image_keys:
			paste_image(item)


		## print the image name and save the image
		print(options.assemble_text+"_out.jpg")
		new_image.save(options.assemble_text+"_out.jpg")

		print("Image saved, clearing thumbnail images")


		## destroy the thumbnail image directory
		shutil.rmtree(os.getcwd()+os.sep+args[0]+"_thumbnail_images")


