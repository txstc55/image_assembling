import json
import ast
from PIL import Image
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
parser = OptionParser()
parser.add_option("-o", "--overlap", dest="overlap", default = False,
                  help="if images can overlap, if false, then images will cut to equal squares")
parser.add_option("-c", "--cut", dest="cut", default = 10,
                  help="the pixel square size you want for the output image")
parser.add_option("-e", "--enlarge", dest="enlarge", default = 10,
                  help="how much you want to enlarge each square")
# parser.add_option("-r", "--repeat", dest="repeat", default = False, 
# 				  help = "if the pictures can be used repeatedly")


(options, args) = parser.parse_args()

if options.overlap == "False" or options.overlap == False:
	options.overlap = False
else:
	options.overlap = True
print(options)
print(args)

random.seed(926)

image_file = args[0]
square_size = int(options.cut)
enlarge_size = int(options.enlarge)

# test_output_img = "thumb.jpg"



current_folder = os.getcwd()
thumbnail_dictionary_str = json.load(open(current_folder+ os.sep+"data"+os.sep+ args[1]+"_data.json", 'r'))
# print(thumbnail_dictionary)

thumbnail_dictionary = {}

if not os.path.exists(os.getcwd()+os.sep+args[1]+"_thumbnail_images"):
    os.makedirs(os.getcwd()+os.sep+args[1]+"_thumbnail_images")



for key in thumbnail_dictionary_str.keys():
	# print(key)
	tuple_key = ast.literal_eval(key)
	thumbnail_dictionary[tuple_key] = thumbnail_dictionary_str[key]



def find_closest_rgb(pixel_value):
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






def image_to_colorblock(image_file, square_size):
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





color_blocks = image_to_colorblock(image_file, square_size)


manager = Manager()
block_image = manager.dict()

# list comprehension to get all the positions
positions = list(itertools.product(list(range(len(color_blocks))), list(range(len(color_blocks[0])))))



# close_color = manager.dict()

def choose_image(pos):
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
	save_thumbnail_image_path = os.getcwd()+os.sep+args[1]+"_thumbnail_images"+os.sep+image_name.split(".")[0]+"_thumb_"+str(square_size*enlarge_size)+ str(options.overlap) +"."+image_name.split(".")[-1]

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
	img = np.uint8(0.75*img+0.25*origin_fragmant)
	img = Image.fromarray(img)
	equalized_image = os.getcwd()+os.sep+args[1]+"_thumbnail_images"+os.sep+image_name.split(".")[0]+"_thumb_"+str(square_size*enlarge_size)+"_"+str(pos)+"."+image_name.split(".")[-1]
	while (not os.path.isfile(equalized_image)):
		try:	
			# scipy.misc.toimage(img, cmin=0.0, cmax=255).save(equalized_image)
			img.save(equalized_image)
		except:
			pass
	block_image[pos] = equalized_image
	print(equalized_image)
	

# choose_image((276, 211))

# print(positions)

pool = Pool(cpu_count())
print("Start multiprocessing")
pool.map(choose_image, positions)
# pool.join()

print("done processing")


# print("creating new image and shared array")
new_image = Image.new("RGB", (len(color_blocks[0])*square_size*enlarge_size, len(color_blocks)*square_size*enlarge_size))
# new_gif = new_image
print(new_image.size)
# new_image = np.ctypeslib.as_ctypes(np.array(Image.new("RGB", (len(color_blocks)*square_size*enlarge_size, len(color_blocks[0])*square_size*enlarge_size))))
# print("fsdfdasf")
# shared_array = sharedctypes.RawArray(new_image._type_, new_image)
# print("done creating")




gif_frames = [new_image]

def paste_image(item):
	x_pos = item[1]
	y_pos = item[0]
	print(item)
	im = Image.open(block_image[item])
	# tmp = np.ctypeslib.as_array(shared_array)
	# tmp[item[0]*square_size*enlarge_size:item[0]*square_size*enlarge_size+square_size*enlarge_size, item[1]*square_size*enlarge_size:item[1]*square_size*enlarge_size+square_size*enlarge_size, 0:3] = np.array(im)
	new_image.paste(im, (x_pos*square_size*enlarge_size, y_pos*square_size*enlarge_size))
	gif_frames.append(new_image)



block_image_keys = block_image.keys()
random.shuffle(block_image_keys)
for item in block_image_keys:
	paste_image(item)



print("".join(args[0].split(".")[:-1])+"_out.jpg")
new_image.save("".join(args[0].split(".")[:-1])+"_out.jpg")

print("Image saved, clearing thumbnail images")



shutil.rmtree(os.getcwd()+os.sep+args[1]+"_thumbnail_images")








# print(image_to_colorblock(image_file, square_size))
