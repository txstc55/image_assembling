# import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
import math
import PIL.ExifTags as ExifTags
from itertools import groupby
from operator import itemgetter
from functools import reduce
from multiprocessing import Pool, cpu_count, Manager
from os import walk
import os
import json
import sys
from optparse import OptionParser


# parser = OptionParser()
# parser.add_option("-o", "--origin", dest="origin", default = False,
#                   help="if true, the rgb information will cover the entire image instead of just the middle part")

# if options.origin == "False" or options.origin == False:
# 	options.origin = False
# else:
# 	options.origin = True




def my_reduce(obj1, obj2):
    return (obj1[0]+obj2[0], obj1[1])



def process_images(image_file):
	# pre process image, store the main n rgb value as the key of this picture
	# print("start")
	print(image_file+" read")
	try:
		if (image_file.lower().endswith("png") or image_file.lower().endswith("jpg")):

			im = Image.open(image_file)
			size = width, height = im.size
			width = width-width%100
			height = height - height%100

			g = math.gcd(width, height)

			width = int(width/g)
			height = int(height/g)

			if width/height>3 or height/width>3:
				return
			else:
				if width<=10 and height<=10:
					width = width*100
					height = height*100
				else:
					width *=10
					height *=10

			
			orient = -1
			try:
				exif=dict((ExifTags.TAGS[k], v) for k, v in im._getexif().items() if k in ExifTags.TAGS)
				orient = exif["Orientation"]
			except:
				pass

			


			# print(width, height)

			im = im.resize((width, height), Image.LANCZOS)
			im = im.filter(ImageFilter.GaussianBlur(radius=2))
			# im.show()

			# if not options.origin:
			if width>height:
				# print("something")
				im = im.crop(((width-height)/2, 0, width - (width-height)/2, height))
			elif height>width:
				im = im.crop((0, (height-width)/2, width, height - (height-width)/2))
			else:
				pass

			# im.show()

			if orient == 3:
				im = im.rotate(180, expand = True)
			elif orient == 6:
				im = im.rotate(270, expand = True)
			elif orient == 8:
				im = im.rotate(90, expand = True)

			width, height = im.size

			
			# im.show()

			local_dic = {}

			colors = im.getcolors(width*height)
			colors = [(x, [z for z in y]) for x, y in colors]
			colors = sorted(colors, key = lambda x: int("".join([str(z) for z in x[1]])))
			colors = [reduce(my_reduce, group) for _, group in groupby(sorted(colors), key=itemgetter(1))][::-1]
			

			for item in colors:
				# print(item)
				color_key = tuple(item[1])
				count = item[0]
				# if count<10:
				# 	break
				if_neighbor = False
				for key in local_dic.keys():
					# print([(a-b)**2 for a, b in zip(key, color_key)])
					# print(sum([(a-b)**2 for a, b in zip(key, color_key)])**(0.5))
					if abs(key[0] - color_key[0])<20 and abs(key[1] - color_key[1])<20 and abs(key[2] - color_key[2])<20 and np.var([key[0] - color_key[0], key[1] - color_key[1], key[2] - color_key[2]])<50:
						local_dic[key]+=count
						if_neighbor = True
						break


				if not if_neighbor:
					local_dic[color_key] = count

			# print(local_dic)



			colors = sorted([(x[:3], local_dic[x]) for x in local_dic], key = lambda x: x[1], reverse=True)
			return_colors = [colors[0]]
			for i in range(1, len(colors)):
				if colors[0][1] - colors[i][1] <0.2* colors[0][1]:
					return_colors.append(colors[i])
				else:
					break
			# print(colors)
			# print(len(colors))
			# print(colors)
			print(image_file+" done processing")
			# print(image_thumbnail_dictionary)
			return return_colors


			# print(colors)
	except:
		print(image_file+" error processing")




		# colors_v = list(filter(lambda x: sum([(y-min(x[1]))**2 for y in x[1]])>200, colors))
		# colors_gray = list(filter(lambda x: sum([(y-min(x[1]))**2 for y in x[1]])<=200, colors))
		# print(height*width)
		# print(colors)
		# print(colors[-1:-21:-1])


# process_images("IMG_2920.jpg")

# print(image_thumbnail_dictionary)




def find_pictures(image_directory):
	f = []
	for (dirpath, dirnames, filenames) in walk(image_directory):
		f.extend(filenames)

	base_path = os.path.basename(image_directory)
	f = [base_path+"/"+x for x in f]
	return f


def store_thumbnails(image_file, thumbnail_dictionary):
	colors = process_images(image_file)
	# print(colors)
	if colors:
		# print(colors)
		for item in colors:

			if item[0] in thumbnail_dictionary:
				thumbnail_dictionary[item[0]]+= [image_file]
			else:
				thumbnail_dictionary[item[0]] = [image_file]

if __name__ == "__main__":
	manager = Manager()
	image_thumbnail_dictionary = manager.dict()

	picture_folder_name = sys.argv[1]
	current_folder = os.getcwd()
	base_folder_name = os.path.basename(picture_folder_name)

	data_json_file = os.path.join(current_folder+ os.sep+ "data"+ os.sep + str(base_folder_name+"_data.json"))

	f= find_pictures(picture_folder_name)
	# print(f)


			# print(image_thumbnail_dictionary)


	pool = Pool(cpu_count())



	print("Start multiprocessing")
	if not os.path.exists(os.path.join(current_folder+ os.sep+ "data"+ os.sep)):
	    os.makedirs(os.path.join(current_folder+ os.sep+ "data"+ os.sep))



	try:
		pool.starmap(store_thumbnails, [(x, image_thumbnail_dictionary) for x in f])
		json_dictionary = {}
		for key in image_thumbnail_dictionary.keys():
			json_dictionary[str(key)] = image_thumbnail_dictionary[key]

		with open(data_json_file, 'w') as d:
			json.dump(json_dictionary.copy(), d)

	except:
		json_dictionary = {}
		for key in image_thumbnail_dictionary.keys():
			json_dictionary[str(key)] = image_thumbnail_dictionary[key]
		with open(data_json_file, 'w') as d:
			json.dump(json_dictionary.copy(), d)



